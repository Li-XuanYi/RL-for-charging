"""Nonnegative charging actions, original state masks, and paper reward terms.

No perfect-model look-ahead or replacement of the requested current is used.
The 4.2 V/309 K limits are measured and penalized. A separate 4.5 V/330 K
numerical stop prevents invalid extrapolation; reaching it fails the episode.
"""
import os
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY','true')
import numpy as np
import pybamm
from model import parameter_values, values, MESH

ACTIONS=np.arange(0.,7.50001,.5)
DT,BETA,TARGET,VMAX,TMAX=90.,.02,.9,4.2,309.


class Cell:
    def __init__(self,vector,soc):
        p=parameter_values(vector)
        p['Upper voltage cut-off [V]']=VMAX
        p['Current function [A]']='[input]'
        p.set_initial_stoichiometries(0.)
        self.c0=float(p['Initial concentration in negative electrode [mol.m-3]'])
        p.set_initial_stoichiometries(1.)
        self.c1=float(p['Initial concentration in negative electrode [mol.m-3]'])
        p.set_initial_stoichiometries(float(soc))
        # Keep the SOC definition tied to the physical 4.2 V reference. The
        # relaxed ceiling below only keeps soft-constraint violations observable.
        p['Upper voltage cut-off [V]']=4.5
        model=pybamm.lithium_ion.SPMe({'thermal':'lumped'})
        model.events.append(pybamm.Event('Numerical temperature ceiling',330-model.variables['X-averaged cell temperature [K]']))
        self.sim=pybamm.Simulation(model,parameter_values=p,
            solver=pybamm.CasadiSolver(mode='safe',rtol=1e-5,atol=1e-7),var_pts=MESH)
        self.initial=self.sim.solve([0,1e-6],inputs={'Current function [A]':0.}).last_state
        self.functions={key:self.initial[key].base_variables_casadi[0] for key in
            ['Voltage [V]','X-averaged cell temperature [K]','R-averaged negative particle concentration [mol.m-3]']}
        self.mapped={}
        self.reset()

    def read(self,sol,key):
        # Evaluate the same compiled PyBaMM expression directly. Reusing its
        # mapped function avoids reserializing it for every 90-second sample.
        # Tested against PyBaMM ProcessedVariable.entries on changing currents.
        n=len(sol.t);cache_key=(key,n)
        if cache_key not in self.mapped:self.mapped[cache_key]=self.functions[key].map(n)
        current=np.asarray(sol.all_inputs[0]['Current function [A]']).reshape(1,1)
        data=self.mapped[cache_key](np.asarray(sol.t).reshape(1,-1),sol.y,np.repeat(current,n,axis=1))
        return np.asarray(data).reshape(-1,n).mean(axis=0)

    def reset(self):
        self.sol=self.initial
        self.update()

    def update(self):
        self.voltage=float(self.read(self.sol,'Voltage [V]')[-1])
        self.temperature=float(self.read(self.sol,'X-averaged cell temperature [K]')[-1])
        c=float(self.read(self.sol,'R-averaged negative particle concentration [mol.m-3]')[-1])
        self.soc=(c-self.c0)/(self.c1-self.c0)

    def propose(self,current,duration=DT):
        if current<0 or current>7.5:
            raise ValueError('MBA-RL current must be in [0,7.5] A')
        return self.sim.solver.step(self.sol,self.sim.built_model,duration,npts=10,save=False,
                                    inputs={'Current function [A]':-float(current)})

    def commit(self,sol,current,duration):
        if duration<=1e-9:
            return dict(current_A=0.,requested_current_A=float(current),duration_s=0.,mean_voltage_V=self.voltage,
                        max_voltage_V=self.voltage,max_temperature_K=self.temperature)
        voltage=self.read(sol,'Voltage [V]');temp=self.read(sol,'X-averaged cell temperature [K]')
        self.sol=sol.last_state;self.update()
        return dict(current_A=float(current),requested_current_A=float(current),duration_s=float(duration),
            mean_voltage_V=float(np.trapz(voltage,sol.t)/(sol.t[-1]-sol.t[0])),
            max_voltage_V=float(voltage.max()),max_temperature_K=float(temp.max()))


class Pack:
    def __init__(self,vectors,initial):
        self.cells=[Cell(v,s) for v,s in zip(vectors,initial)]
        self.reset()

    def reset(self):
        for c in self.cells:c.reset()
        self.time=0.

    def physical(self):
        return np.array([[c.soc,c.voltage,c.temperature] for c in self.cells])

    def obs(self):
        return ((self.physical()-[.5,3.5,308.])/[.5,1.,11.]).astype(np.float32)

    def mask(self):
        mask=np.ones((3,len(ACTIONS)),dtype=bool)
        for i,c in enumerate(self.cells):
            if c.soc>=.95:mask[i,1:]=False
            elif c.voltage>=VMAX or c.temperature>=TMAX:mask[i,ACTIONS>1.]=False
        return mask

    def step(self,actions):
        trials=[];failure=False;reason=''
        try:
            trials=[c.propose(a) for c,a in zip(self.cells,actions)]
            durations=[float(s.t[-1]-c.sol.t[-1]) for c,s in zip(self.cells,trials)]
            duration=min(durations)
            failure=duration<DT-1e-3
            if failure:
                reason='; '.join(str(s.termination) for s in trials if str(s.termination)!='final time')
                # End ALL cells at the same physical instant. Do not leave the
                # other cells 90 seconds ahead of the cell that stopped early.
                if duration<=1e-9:trials=[c.sol for c in self.cells]
                else:
                    for i,(c,a) in enumerate(zip(self.cells,actions)):
                        if durations[i]>duration+1e-4:trials[i]=c.propose(a,duration)
                    endpoints=[s.t[-1] for s in trials]
                    if max(endpoints)-min(endpoints)>1e-3:
                        failure=True;duration=0.;reason='Inconsistent event timing; discarded attempted interval'
                        trials=[c.sol for c in self.cells]
        except pybamm.SolverError as error:
            failure=True;duration=0.;reason=str(error)[:500]
            trials=[c.sol for c in self.cells]
        stats=[c.commit(s,a,duration) for c,s,a in zip(self.cells,trials,actions)]
        self.time+=duration
        x=self.physical();std=float(x[:,0].std())
        r_time=-.75
        r_bal=-50*max(0.,std-BETA)
        r_volt=-20*np.maximum(x[:,1]-VMAX,0).sum()
        r_temp=-2*np.maximum(x[:,2]-TMAX,0).sum()
        # Missing from the supplied code: numerical failures are terminal and
        # must not be rewarded for avoiding future time/imbalance penalties.
        # -2000 is worse than any 50-step trajectory respecting the soft safety
        # limits and SOC in [0,1]. This reconstruction assumption is logged.
        r_failure=-2000. if failure else 0.
        reward=float(r_time+r_bal+r_volt+r_temp+r_failure)
        goal=bool(std<=BETA and x[:,0].min()>=TARGET)
        return reward,bool(goal or failure),stats,dict(time=r_time,balance=r_bal,voltage=float(r_volt),temperature=float(r_temp),
            numerical_failure=failure,failure_reason=reason,failure_penalty=r_failure,goal=goal)

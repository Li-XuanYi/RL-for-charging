"""Independent output checks, including a fresh static PyBaMM fit solve."""
import os
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY','true')
import argparse,csv,json
from pathlib import Path
import numpy as np
import pybamm
import torch
from model import parameter_values,values,MESH,FitSimulator
from battery import Pack
from learner import Learner
from run_mainline import rollout,SCENARIOS


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
    fitted=json.loads((run/'calibration/parameters.json').read_text(encoding='utf-8'))
    comparisons=[]
    cache=FitSimulator();cache.sim.solver.rtol=1e-8;cache.sim.solver.atol=1e-10
    for number,vector in enumerate(fitted['vectors'],1):
        data=np.loadtxt(run/'calibration'/f'battery_{number}_fit.csv',delimiter=',',skiprows=1)
        params=parameter_values(vector,float(data[0,3]+273.15));params['Current function [A]']=fitted['current_A_assumption']
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe({'thermal':'lumped'}),parameter_values=params,
            solver=pybamm.CasadiSolver(mode='safe',rtol=1e-8,atol=1e-10),var_pts=MESH)
        sol=sim.solve(data[:,0])
        v=np.interp(data[:,0],sol.t,values(sol,'Voltage [V]'))
        temp=np.interp(data[:,0],sol.t,values(sol,'X-averaged cell temperature [K]')-273.15)
        dv=float(np.max(np.abs(v-data[:,2])));dt=float(np.max(np.abs(temp-data[:,4])))
        ct,cv,ctemp=cache.solve(vector,fitted['current_A_assumption'],data[-1,0],data[0,3]+273.15,points=len(data))
        same_v=float(np.max(abs(v-np.interp(data[:,0],ct,cv))))
        same_t=float(np.max(abs(temp-np.interp(data[:,0],ct,ctemp))))
        assert same_v<1e-5 and same_t<1e-4,(number,same_v,same_t)
        # Saved fits use rtol=1e-5, atol=1e-7. Record numerical sensitivity
        # separately from the strict same-tolerance implementation comparison.
        assert dv<.002 and dt<.1,(number,dv,dt)
        comparisons.append(dict(cell=number,tight_static_vs_cached_voltage_difference_V=same_v,
            tight_static_vs_cached_temperature_difference_C=same_t,
            saved_fit_vs_tight_voltage_difference_V=dv,saved_fit_vs_tight_temperature_difference_C=dt,
            tight_voltage_RMSE_V=float(np.mean((v-data[:,1])**2)**.5),
            tight_temperature_RMSE_C=float(np.mean((temp-data[:,3])**2)**.5)))
    # At 50-episode boundaries the original-network target should have copied.
    target_checks=[]
    for checkpoint in (run/'checkpoints').glob('*_50.pt'):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,ck['target_agent'][k]) for k,v in ck['agent'].items())
        assert all(torch.equal(v,ck['target_mixer'][k]) for k,v in ck['mixer'].items())
        target_checks.append(checkpoint.name)
    summary=list(csv.DictReader((run/'summary.csv').open(encoding='utf-8-sig')))
    checked=[];replay_checks=[]
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    for record in summary:
        path=run/'trajectories'/('final_'+record['scenario']+'.csv')
        rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        keys=['time_s','soc','soc_std','current_A','interval_max_voltage_V','interval_max_temperature_K']
        x=np.array([[float(r[k]) for k in keys] for r in rows]).reshape(-1,3,6)
        assert np.isfinite(x).all()
        assert np.allclose(x[:,:,0],x[:,0:1,0])
        durations=np.diff(x[:,0,0]);assert (durations>=-1e-7).all() and (durations<=90+1e-3).all()
        if record['numerical_failure']=='False':assert np.allclose(durations,90)
        assert np.allclose(x[:,:,1].std(axis=1),x[:,0,2])
        assert (x[:,:,3]>=0).all() and (x[:,:,3]<=7.5).all()
        assert int((x[1:,:,4]>4.2+1e-5).sum())==int(record['voltage_violating_cell_intervals'])
        assert int((x[1:,:,5]>309+1e-5).sum())==int(record['temperature_violating_cell_intervals'])
        target=((x[:,:,1].min(axis=1)>=.9)&(x[:,0,2]<=.02)).any()
        assert bool(target)==(record['soc_target_reached']=='True')
        learner=Learner(config['seed'])
        learner.load(run/'checkpoints'/f"{record['scenario']}_{config['episodes']}.pt")
        pack=Pack(fitted['vectors'],SCENARIOS[record['scenario']])
        _,replayed=rollout(pack,learner,0.,60)
        replay_values=np.array([[float(r[k]) for k in keys] for r in replayed]).reshape(-1,3,6)
        assert replay_values.shape==x.shape
        error=float(np.max(np.abs(replay_values-x)))
        assert np.allclose(replay_values,x,rtol=1e-7,atol=1e-6),(record['scenario'],error)
        replay_checks.append(dict(scenario=record['scenario'],max_difference=error))
        checked.append(path.name)
    assert len(checked)==2 and len(target_checks)==(2 if config['episodes']>=50 else 0)
    report=dict(status='passed',static_model_checks=comparisons,target_sync_checks=target_checks,trajectory_checks=checked,checkpoint_replay_checks=replay_checks,
        solver_tolerance_note='Saved fits: rtol 1e-5/atol 1e-7. Independent static and cached comparison: 1e-8/1e-10. Saved-fit sensitivity limits: 0.002 V/0.1 C; same-tight-tolerance implementation limits: 1e-5 V/1e-4 C.',
        meaning='Implementation and data consistency checks; not proof of convergence or exact-paper replication')
    (run/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

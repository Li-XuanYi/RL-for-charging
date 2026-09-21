"""Battery model following supplied SPMe code and recorded parameter dictionary."""
import os
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY','true')
import json
from pathlib import Path
import numpy as np
import pybamm

BASE=json.loads((Path(__file__).parent/'original_parameters.json').read_text())
KEYS=['Positive electrode diffusivity [m2.s-1]', 'Negative electrode diffusivity [m2.s-1]',
      'Positive particle radius [m]', 'Negative particle radius [m]',
      'Negative electrode active material volume fraction','Positive electrode active material volume fraction']
SCALES=np.array([1e-15,1e-14,1e-6,1e-6,.1,.1])
HEAT_KEYS=[n+' specific heat capacity [J.kg-1.K-1]' for n in
           ['Negative current collector','Positive current collector','Negative electrode','Positive electrode','Separator']]
INITIAL=np.r_[np.array([BASE[k] for k in KEYS])/SCALES,25.,1.]
# Enclose all hardcoded original values, including the original n-fraction .56
# which was incorrectly outside the supplied PSO's [.60,.65] clamp.
BOUNDS=np.array([[5.,6.],[4.,5.],[3.8,5.],[3.8,5.],[4.5,7.5],[4.,6.6],[5.,50.],[.4,2.5]])
MESH={'x_n':30,'x_s':30,'x_p':30,'r_n':10,'r_p':10}
FREE=np.array([0,1,4,5,6,7])


def parameter_values(vector=None,temperature=298.15):
    p=pybamm.ParameterValues('Chen2020')
    p.update(BASE)
    p.update({'Ambient temperature [K]':298.15,'Initial temperature [K]':float(temperature),
              'Current function [A]':0.})
    if vector is not None:
        p.update(dict(zip(KEYS,np.asarray(vector[:6])*SCALES)))
        p['Total heat transfer coefficient [W.m-2.K-1]']=float(vector[6])
        for k in HEAT_KEYS:
            p[k]=BASE[k]*float(vector[7])
    return p


def values(sol,key):
    return np.asarray(sol[key].entries).reshape(-1,len(sol.t)).mean(axis=0)


class FitSimulator:
    def __init__(self):
        p=parameter_values()
        for k in [KEYS[i] for i in (0,1,4,5)]+HEAT_KEYS+['Total heat transfer coefficient [W.m-2.K-1]','Current function [A]','Initial temperature [K]']:
            p[k]='[input]'
        self.sim=pybamm.Simulation(pybamm.lithium_ion.SPMe({'thermal':'lumped'}),parameter_values=p,
            solver=pybamm.CasadiSolver(mode='safe',rtol=1e-5,atol=1e-7),var_pts=MESH)
        self.sim.build()

    def solve(self,vector,current,duration,temperature=298.15,points=301):
        if not np.allclose(vector[2:4],INITIAL[2:4],atol=1e-12):
            raise ValueError('Particle radii are fixed to original source values in the cached model')
        inputs={KEYS[i]:float(vector[i]*SCALES[i]) for i in (0,1,4,5)}
        inputs.update({'Current function [A]':float(current),'Initial temperature [K]':float(temperature),
                       'Total heat transfer coefficient [W.m-2.K-1]':float(vector[6])})
        inputs.update({k:BASE[k]*float(vector[7]) for k in HEAT_KEYS})
        # The original fitting script preserves Chen2020's initial concentrations.
        # Do not silently reset them with initial_soc=1.
        sol=self.sim.solve(np.linspace(0,duration,points),inputs=inputs)
        return sol.t,values(sol,'Voltage [V]'),values(sol,'X-averaged cell temperature [K]')-273.15

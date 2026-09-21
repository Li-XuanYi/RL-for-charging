import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pybamm
from model import FitSimulator, INITIAL, BOUNDS, FREE


def load_data(root):
    directory=next(root.glob('multi-balancing-agent*/multi-balancing-agent*/data_true/data_image'))
    result=[]
    for name in ['begin','middle','age']:
        v=np.loadtxt(directory/('data_1C_'+name))
        t=np.loadtxt(directory/('temp_data 1C '+name))
        if len(v)!=len(t) or not np.isfinite(v).all() or not np.isfinite(t).all():
            raise ValueError('Invalid measured pair: '+name)
        result.append((np.arange(len(v),dtype=float),v,t))
    return result


def calibrate(root,out,current=5.,particles=20,iterations=35,seed=7):
    out.mkdir(exist_ok=True,parents=True)
    simulator=FitSimulator()
    all_vectors,metrics=[],[]
    low,high=BOUNDS[FREE].T
    for number,(t,v,temp) in enumerate(load_data(root),1):
        rng=np.random.default_rng(seed+number)
        position=rng.uniform(low,high,size=(particles,len(FREE)))
        position[0]=INITIAL[FREE]
        # Warm candidates scale active-material fractions by the measured
        # discharge durations relative to the original aged-cell parameter set.
        for j in range(1,min(particles,5)):
            candidate=INITIAL.copy()
            ratio=t[-1]/2676.
            candidate[4:6]*=ratio
            candidate[6:]*=np.exp(rng.normal(0,.15,2))
            position[j]=np.clip(candidate[FREE],low,high)
        velocity=np.zeros_like(position)
        personal=position.copy()
        personal_scores=np.full(particles,np.inf)
        best,best_score=position[0].copy(),np.inf
        history=[]
        failures=0
        def objective(z):
            nonlocal failures
            vec=INITIAL.copy(); vec[FREE]=z
            try:
                st,sv,stemp=simulator.solve(vec,current,t[-1],temp[0]+273.15)
                err_v=np.sqrt(np.mean((np.interp(t,st,sv)-v)**2))
                err_t=np.sqrt(np.mean((np.interp(t,st,stemp)-temp)**2))
                return float((err_v/.03)**2+(err_t/.3)**2+100*max(0.,1-st[-1]/t[-1])**2)
            except (pybamm.SolverError,ValueError):
                failures+=1
                return 1e8
        for iteration in range(iterations+1):
            for k in range(particles):
                score=objective(position[k])
                if score<personal_scores[k]:
                    personal_scores[k],personal[k]=score,position[k].copy()
                if score<best_score:
                    best_score,best=score,position[k].copy()
            vec=INITIAL.copy();vec[FREE]=best
            history.append([iteration,best_score])
            (out/f'battery_{number}_progress.json').write_text(json.dumps({'iteration':iteration,'objective':best_score,'vector':vec.tolist()}),encoding='utf-8')
            if iteration%5==0 or iteration==iterations:
                print(f'Calibration cell {number} {iteration}/{iterations}: objective={best_score:.5f}',flush=True)
            w=.9-.5*iteration/max(1,iterations)
            velocity=w*velocity+1.5*rng.random(position.shape)*(personal-position)+1.5*rng.random(position.shape)*(best-position)
            position=np.clip(position+velocity,low,high)
        if best_score>=1e8:
            raise RuntimeError('All calibration evaluations failed')
        vec=INITIAL.copy();vec[FREE]=best
        st,sv,stemp=simulator.solve(vec,current,t[-1],temp[0]+273.15,points=len(t))
        pred_v=np.interp(t,st,sv);pred_t=np.interp(t,st,stemp)
        record={'cell':number,'voltage_RMSE_V':float(np.mean((v-pred_v)**2)**.5),
                'temperature_RMSE_C':float(np.mean((temp-pred_t)**2)**.5),
                'model_end_s':float(st[-1]),'measured_end_s':float(t[-1]),
                'current_A_assumption':current,'solver_failed_candidates':failures,
                'evaluation':'in_sample_fit_not_independent_validation'}
        metrics.append(record);all_vectors.append(vec.tolist())
        np.savetxt(out/f'battery_{number}_fit.csv',np.column_stack([t,v,pred_v,temp,pred_t,t<=st[-1]]),delimiter=',',
                   header='time_s,measured_V,fitted_V,measured_C,fitted_C,model_covers_time',comments='')
        np.savetxt(out/f'battery_{number}_pso.csv',history,delimiter=',',header='iteration,objective',comments='')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,actual,predicted,label in zip(axes,[v,temp],[sv,stemp],['Voltage / V','Temperature / deg C']):
            ax.plot(t,actual,label='Measured',lw=1.3);ax.plot(st,predicted,label='Refitted original SPMe',lw=1.3)
            ax.set(xlabel='Time / s',ylabel=label);ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'Cell {number} | original-code current assumption {current:g} A | in-sample fit')
        fig.tight_layout();fig.savefig(out/f'battery_{number}_fit.png',dpi=180);plt.close(fig)
        print('Fit result:',json.dumps(record),flush=True)
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        (out/'parameters.json').write_text(json.dumps({'vectors':all_vectors,'current_A_assumption':current,
            'source':'original_parameter_dictionary_plus_reidentified_coefficients','fixed_particle_radii_m':[3.98e-6,4.18e-6],
            'assumptions':['default initial concentrations during discharge, following original fitting code',
            'source uses 5 A but paper states 4 Ah at 1 C; actual experimental current is unverified',
            'particle radii fixed to supplied parameter values; fit diffusion, active fractions, heat transfer and heat-capacity scale',
            'search bounds include original hardcoded parameters which were outside original PSO bounds']},indent=2),encoding='utf-8')
    return all_vectors,metrics


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--current',type=float,default=5.);p.add_argument('--iterations',type=int,default=35)
    args=p.parse_args();calibrate(args.root,args.out,args.current,iterations=args.iterations)

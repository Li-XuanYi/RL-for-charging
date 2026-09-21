"""Measured discharge -> refitted original SPMe -> QMIX -> paper metrics."""
import os
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY','true')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,csv,hashlib,json,random,sys,time,traceback
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
from calibrate import calibrate,load_data
from battery import Pack,ACTIONS,DT,BETA,VMAX,TMAX
from learner import Learner

SCENARIOS={'A':[.1,.2,.3],'B':[.3,.5,.7]}


def dump(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)


def csv_write(path,rows):
    if rows:
        with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Tee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,text):self.stream.write(text);self.log.write(text);self.log.flush()
    def flush(self):self.stream.flush();self.log.flush()


def rows_for(pack,stats=None,components=None):
    std=float(pack.physical()[:,0].std())
    result=[]
    for i,c in enumerate(pack.cells):
        s=stats[i] if stats else dict(current_A=0.,requested_current_A=0.,duration_s=0.,mean_voltage_V=c.voltage,max_voltage_V=c.voltage,max_temperature_K=c.temperature)
        result.append(dict(time_s=pack.time,cell=i+1,soc=c.soc,soc_std=std,voltage_V=c.voltage,
            temperature_K=c.temperature,current_A=s['current_A'],requested_current_A=s['requested_current_A'],interval_duration_s=s['duration_s'],interval_mean_voltage_V=s['mean_voltage_V'],
            interval_max_voltage_V=s['max_voltage_V'],interval_max_temperature_K=s['max_temperature_K'],
            reward_time=components['time'] if components else 0.,reward_balance=components['balance'] if components else 0.,
            reward_voltage=components['voltage'] if components else 0.,reward_temperature=components['temperature'] if components else 0.,
            numerical_failure=components['numerical_failure'] if components else False,
            failure_reason=components['failure_reason'] if components else '',failure_penalty=components['failure_penalty'] if components else 0.))
    return result


def rollout(pack,learner,epsilon,limit=50):
    pack.reset();learner.reset()
    ep=dict(obs=[pack.obs()],masks=[pack.mask()],actions=[],rewards=[],done=[],goals=[],failures=[])
    rows=rows_for(pack)
    for step in range(limit):
        choices=learner.choose(ep['obs'][-1],ep['masks'][-1],epsilon)
        reward,done,stats,parts=pack.step(ACTIONS[choices])
        ep['actions'].append(choices);ep['rewards'].append(reward);ep['done'].append(done)
        ep['goals'].append(parts['goal']);ep['failures'].append(parts['numerical_failure'])
        ep['obs'].append(pack.obs());ep['masks'].append(pack.mask())
        rows.extend(rows_for(pack,stats,parts))
        if done:break
    return ep,rows


def metrics(rows,scenario,episode):
    def array(key):return np.array([r[key] for r in rows]).reshape(-1,3)
    t=array('time_s')[:,0];soc=array('soc');std=soc.std(axis=1)
    balance=std<=BETA;target=balance&(soc.min(axis=1)>=.9)
    first=int(np.flatnonzero(balance)[0]) if balance.any() else None
    vm=array('interval_max_voltage_V');tm=array('interval_max_temperature_K')
    vviol=vm[1:]>VMAX+1e-5;tviol=tm[1:]>TMAX+1e-5
    current=array('current_A')[1:];meanv=array('interval_mean_voltage_V')[1:];duration=array('interval_duration_s')[1:]
    failed=any(r['numerical_failure'] for r in rows)
    constraints=not bool(vviol.any() or tviol.any() or failed)
    return dict(scenario=scenario,training_episode=episode,first_balance_s=float(t[first]) if first is not None else None,
        soc_target90_s=float(t[np.flatnonzero(target)[0]]) if target.any() else None,
        soc_target_reached=bool(target.any()),balance_maintained_after_first=bool(balance[first:].all()) if first is not None else False,
        max_std_after_first=float(std[first:].max()) if first is not None else None,
        final_soc_1=float(soc[-1,0]),final_soc_2=float(soc[-1,1]),final_soc_3=float(soc[-1,2]),final_soc_std=float(std[-1]),
        max_voltage_V=float(vm.max()),max_temperature_K=float(tm.max()),
        voltage_violating_cell_intervals=int(vviol.sum()),temperature_violating_cell_intervals=int(tviol.sum()),
        constraints_satisfied=constraints,charging_task_success=bool(target.any() and constraints),
        numerical_failure=failed,total_charged_Ah=float((current*duration).sum()/3600),terminal_input_Wh=float((current*meanv*duration).sum()/3600),
        lookahead_filter_interventions=0,final_time_s=float(t[-1]))


def plot_trajectory(rows,out,scenario):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i in (1,2,3):
        cell=[r for r in rows if r['cell']==i];t=[r['time_s'] for r in cell]
        for ax,key,ylabel in zip(axes.flat,['soc','current_A','voltage_V','temperature_K'],['SOC','Current / A','Voltage / V','Temperature / K']):
            if key=='current_A':ax.step(t,[r[key] for r in cell],where='pre',label=f'Cell {i}')
            else:ax.plot(t,[r[key] for r in cell],label=f'Cell {i}')
            ax.set(xlabel='Time / s',ylabel=ylabel);ax.grid(alpha=.2)
    axes[0,0].legend();axes[0,0].axhline(.9,color='gray',ls='--')
    axes[1,0].axhline(VMAX,color='gray',ls='--');axes[1,1].axhline(TMAX,color='gray',ls='--')
    fig.suptitle(f'Restored numerical mainline | Set {scenario} | no look-ahead current replacement')
    fig.tight_layout();fig.savefig(out/f'trajectory_{scenario}.png',dpi=180);plt.close(fig)
    cell=[r for r in rows if r['cell']==1]
    fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['time_s'] for r in cell],[r['soc_std'] for r in cell])
    ax.axhline(BETA,color='gray',ls='--',label='Balance threshold 0.02');ax.legend()
    ax.set(xlabel='Time / s',ylabel='SOC standard deviation',title=f'Set {scenario}: balance and possible recurrence')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'balance_{scenario}.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--particles',type=int,default=20)
    parser.add_argument('--fit-iterations',type=int,default=35)
    parser.add_argument('--current',type=float,default=5.)
    parser.add_argument('--fit-from',type=Path,help='Directory containing parameters.json and fit outputs')
    parser.add_argument('--fit-only',action='store_true')
    parser.add_argument('--eval-every',type=int,default=25)
    args=parser.parse_args()
    if min(args.episodes,args.particles,args.fit_iterations,args.eval_every)<1 or args.current<=0:parser.error('budgets and current must be positive')
    root=args.root.resolve();out=root/'mainline_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for d in ['calibration','training','checkpoints','trajectories','figures']:(out/d).mkdir(parents=True,exist_ok=True)
    log=(out/'run.log').open('w',encoding='utf-8');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    print('OUTPUT:',out,flush=True);started=time.time()
    np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(min(4,os.cpu_count() or 1))
    config={**vars(args),'root':str(root),'fit_from':str(args.fit_from) if args.fit_from else None,
        'scope':'restored_numerical_mainline_not_exact_paper_replication','model':'SPMe with supplied Chen2020-based dictionary',
        'current_assumption':'5 A follows supplied fitting code; 4 A in paper remains unresolved',
        'particle_radii':'fixed to original values; four electrochemical and two thermal coefficients refitted',
        'observation':'SOC, voltage, temperature','actions_A':ACTIONS.tolist(),'decision_s':DT,
        'reward':{'time':-.75,'balance':-50,'voltage':-20,'temperature':-2},
        'control_mask':'original state mask adapted to nonnegative paper actions; zero current at SOC>=.95; <=1 A at measured V>=4.2 or T>=309',
        'lookahead_current_replacement':False,'soft_limits':{'voltage_V':VMAX,'temperature_K':TMAX},
        'numerical_stops':{'voltage_V':4.5,'temperature_K':330,'handling':'terminal failed episode; synchronized actual duration; continue next episode','additional_failure_penalty':-2000},
        'training':{'gamma':.99,'learning_rate':2e-4,'optimizer':'RMSprop','replay':800,'batch':128,'target_sync_episodes':50,'episode_limit':50,'epsilon':'0.5*(1-(epoch+1)/episode_budget)'},
        'evaluation':'greedy on the same two initial SOC scenarios; not an independent generalization test',
        'paper_reference':{'first_balance_s':{'A':630,'B':1170},'voltage_RMSE_V':[.026,.035,.029],'temperature_RMSE':[.068,.091,.093]},
        'hardware':'out_of_scope','DQL':'out_of_scope_for_this_mainline; missing source not replaced with an invented baseline'}
    dump(out/'config.json',config);dump(out/'status.json',{'status':'running'})
    (root/'mainline_results'/'LATEST.txt').write_text(str(out),encoding='utf-8')
    dump(out/'code_provenance.json',[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in Path(__file__).parent.rglob('*.py')])
    summary=[]
    try:
        if args.fit_from:
            import shutil
            fitted=json.loads((args.fit_from/'parameters.json').read_text(encoding='utf-8'))
            if fitted['current_A_assumption']!=args.current:raise ValueError('Reused fit current mismatch')
            vectors=fitted['vectors'];fit_metrics=json.loads((args.fit_from/'metrics.json').read_text(encoding='utf-8'))
            if np.asarray(vectors).shape!=(3,8) or not np.isfinite(vectors).all():raise ValueError('Expected 3x8 finite fit parameters')
            for p in args.fit_from.iterdir():
                if p.is_file():shutil.copy2(p,out/'calibration'/p.name)
        else:
            vectors,fit_metrics=calibrate(root,out/'calibration',args.current,args.particles,args.fit_iterations,args.seed)
        csv_write(out/'calibration'/'metrics.csv',fit_metrics)
        if args.fit_only:
            dump(out/'status.json',{'status':'completed_calibration_only','elapsed_s':time.time()-started});return
        for name,initial in SCENARIOS.items():
            print(f'Train Set {name}, {args.episodes} episodes, initial={initial}',flush=True)
            learner=Learner(args.seed);pack=Pack(vectors,initial);buffer=deque(maxlen=800);history=[];evaluations=[]
            ep,rows=rollout(pack,learner,0.,60)
            evaluations.append(metrics(rows,name,0))
            csv_write(out/'trajectories'/f'untrained_{name}.csv',rows)
            for epoch in range(args.episodes):
                epsilon=.5*(1-(epoch+1)/args.episodes)
                ep,rows=rollout(pack,learner,epsilon)
                buffer.append(ep);loss=None
                if len(buffer)>=5:
                    indices=learner.rng.choice(len(buffer),size=min(len(buffer),128),replace=False)
                    loss=learner.train([buffer[int(i)] for i in indices])
                if (epoch+1)%50==0:learner.sync()
                safe=max(r['interval_max_voltage_V'] for r in rows)<=VMAX+1e-5 and max(r['interval_max_temperature_K'] for r in rows)<=TMAX+1e-5
                history.append(dict(episode=epoch+1,reward=float(sum(ep['rewards'])),loss=loss,epsilon=epsilon,
                    steps=len(ep['actions']),soc_target_reached=ep['goals'][-1],numerical_failure=ep['failures'][-1],constraints_satisfied=bool(safe and not ep['failures'][-1])))
                csv_write(out/'training'/f'{name}.csv',history)
                if (epoch+1)%10==0:print(f'{name} {epoch+1}/{args.episodes}: reward={sum(ep["rewards"]):.2f}, target={ep["goals"][-1]}, safe={safe and not ep["failures"][-1]}',flush=True)
                if (epoch+1)%args.eval_every==0 or epoch+1==args.episodes:
                    learner.save(out/'checkpoints'/f'{name}_{epoch+1}.pt',{'episode':epoch+1,'scenario':name,'config':config})
                    evaluation,trace=rollout(pack,learner,0.,60)
                    result=metrics(trace,name,epoch+1);evaluations.append(result)
                    csv_write(out/'training'/f'{name}_evaluations.csv',evaluations)
                    csv_write(out/'trajectories'/f'{name}_episode_{epoch+1}.csv',trace)
                    print('Greedy evaluation:',json.dumps(result),flush=True)
            learner.load(out/'checkpoints'/f'{name}_{args.episodes}.pt')
            _,rows=rollout(pack,learner,0.,60);result=metrics(rows,name,args.episodes)
            summary.append(result);csv_write(out/'summary.csv',summary)
            csv_write(out/'trajectories'/f'final_{name}.csv',rows)
            plot_trajectory(rows,out/'figures',name)
            fig,ax=plt.subplots(figsize=(8,4));ax.plot([h['episode'] for h in history],[h['reward'] for h in history],alpha=.6)
            ax.set(xlabel='Training episode',ylabel='Team reward',title=f'Set {name} training');ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(out/'figures'/f'training_{name}.png',dpi=180);plt.close(fig)
        dump(out/'status.json',{'status':'completed','elapsed_s':time.time()-started,'scope':config['scope'],
            'fit_metrics':fit_metrics,'results':summary,'all_SOC_targets_reached':all(r['soc_target_reached'] for r in summary),
            'all_charging_tasks_satisfy_constraints':all(r['charging_task_success'] for r in summary)})
        print('Mainline completed. See calibration/metrics.csv, summary.csv and status.json. OUTPUT:',out,flush=True)
    except BaseException as e:
        dump(out/'status.json',{'status':'interrupted' if isinstance(e,KeyboardInterrupt) else 'failed','error':str(e),'elapsed_s':time.time()-started,'partial_results':summary})
        traceback.print_exc();raise


if __name__=='__main__':main()

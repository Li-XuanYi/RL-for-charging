"""Continuous current screen and explicit sampled CC-CV engineering comparators."""
import paths
import numpy as np
import pybamm
from control import SafetyController, _NumericalFailure, VMAX,TMAX,ACTIONS


class ContinuousScreen(SafetyController):
    """Same nominal model/margins/interlock as discrete screen; no current quantization.

    Backtracking fractions form a deterministic finite candidate set. These are
    not a proof of recursive feasibility or a solution of an optimal projection.
    """
    def filter(self,requested,observation):
        requested=np.asarray(requested,dtype=float)
        observation=np.asarray(observation,dtype=float)
        if requested.shape!=(self.nominal.n,) or observation.shape!=(self.nominal.n,3):
            raise ValueError('Continuous screen dimension mismatch')
        if not np.isfinite(requested).all() or not np.isfinite(observation).all() or np.any(requested<0) or np.any(requested>7.5+1e-8):
            raise ValueError('Invalid continuous current or observation')
        nominal=self.nominal.physical()
        bias=np.column_stack([np.abs(observation[:,0]-nominal[:,0]),
                              np.maximum(observation[:,1]-nominal[:,1],0),
                              np.maximum(observation[:,2]-nominal[:,2],0)])
        applied=np.zeros_like(requested)
        details={'infeasible':False,'intervention':False,'cells':[],
                 'candidate_solve_count':0,'reason':'continuous backtracking screen',
                 'requested_current_A':requested.tolist(),'action_quantization':False}
        if self.observer_failed:
            details.update(infeasible=True,reason=self.observer_failure_reason)
            return applied,details
        for i,(cell,command) in enumerate(zip(self.nominal.cells,requested)):
            feasible=False;reject=[]
            candidates=np.unique(np.r_[command*np.array([1.,.875,.75,.625,.5,.375,.25,.125]),0.])[::-1]
            for current in candidates:
                try:
                    details['candidate_solve_count']+=1
                    sol=cell.propose(float(current),self.decision_s)
                    a=cell.trajectory(sol)
                    if str(sol.termination)!='final time' or a['time_s'][-1]<self.decision_s-1e-5:
                        raise _NumericalFailure('Predicted candidate ended early')
                    peaks=np.array([max(a['soc']),max(a['voltage_V']),max(a['temperature_K'])])+bias[i]
                    limits=np.array([self.soc_max,VMAX-self.voltage_margin_V,TMAX-self.temperature_margin_K])
                    if np.any(peaks>limits):
                        reject.append({'current_A':float(current),'peaks':peaks.tolist()});continue
                    details['candidate_solve_count']+=1
                    backup=cell.propose(0.,self.decision_s,old_solution=sol.last_state)
                    b=cell.trajectory(backup,origin=float(sol.t[-1]))
                    if str(backup.termination)!='final time' or b['time_s'][-1]<self.decision_s-1e-5:
                        raise _NumericalFailure('Predicted backup ended early')
                    other=np.array([max(b['soc']),max(b['voltage_V']),max(b['temperature_K'])])+bias[i]
                    if np.any(other>limits):
                        reject.append({'current_A':float(current),'backup_peaks':other.tolist()});continue
                    applied[i]=current;feasible=True;break
                except (pybamm.SolverError,_NumericalFailure) as exc:
                    reject.append({'current_A':float(current),'solver_error':str(exc)[:300]})
            details['cells'].append({'cell':i+1,'feasible':feasible,'rejections':reject})
            if not feasible:
                details['infeasible']=True
        if details['infeasible']:
            applied[:]=0.
            details['reason']='No feasible continuous candidate; zero is not declared safe'
        details['intervention']=bool(not np.allclose(applied,requested,atol=1e-10))
        details['applied_current_A']=applied.tolist()
        return applied,details


class RulePolicy:
    """Observable-state rules with an explicit common terminal SOC target.

    Stopping every cell separately at 0.90 can strand a pack after a sampled
    overshoot: all currents become zero although pack SOC spread is still too
    large.  Both rule comparators therefore request charge toward 0.93, taper
    current as that observed SOC approaches, and leave the common environment
    to stop at the first all-cells >=0.90, std <=0.02 state.  The taper gain is a
    fixed engineering parameter, not a capacity estimate or access to the true
    plant's internal state.  The numerical screen may reduce requests further.
    """
    learning_kind='rule'
    def __init__(self,method,n_agents,seed,config):
        if method not in ('cccv','cccv_continuous','soc_rule'):
            raise ValueError('Unknown rule comparator')
        if isinstance(n_agents,bool) or not isinstance(n_agents,(int,np.integer)) or n_agents<1:
            raise ValueError('Rule n_agents must be a positive integer')
        self.method=method;self.n_agents=n_agents;self.config=dict(config);self.seed=seed
        defaults={'rule_base_A':3.,'rule_gain_A_per_soc':20.,'cc_current_A':7.5,
                  'cv_gain_A_per_V':20.,'voltage_margin_V':.03,'soc_max':.95,
                  'rule_terminal_soc_target':.93,'rule_terminal_taper_A_per_soc':75.,
                  'rule_terminal_min_request_A':.5}
        for key,value in defaults.items():
            self.config.setdefault(key,value)
            if isinstance(self.config[key],bool) or not np.isfinite(self.config[key]):
                raise ValueError(f'Rule parameter {key} must be finite and numeric')
        if not 0<self.config['cc_current_A']<=7.5 or not 0<self.config['rule_base_A']<=7.5:
            raise ValueError('Rule CC and base current must be in (0,7.5] A')
        if self.config['rule_gain_A_per_soc']<0 or self.config['cv_gain_A_per_V']<=0:
            raise ValueError('Rule balance gain must be nonnegative and CV gain positive')
        if not 0<=self.config['voltage_margin_V']<.5:
            raise ValueError('Invalid rule voltage margin')
        if not .9<self.config['rule_terminal_soc_target']<self.config['soc_max']<1:
            raise ValueError('Terminal SOC target must be strictly between 0.90 and physical SOC maximum')
        if self.config['rule_terminal_taper_A_per_soc']<=0:
            raise ValueError('Terminal current taper gain must be positive')
        if not .5<=self.config['rule_terminal_min_request_A']<=7.5:
            raise ValueError('Terminal minimum request must be in [0.5,7.5] A')
        self.action_kind='continuous' if method=='cccv_continuous' else 'discrete'
        self.reset()
    def reset(self):
        self.cv=np.zeros(self.n_agents,dtype=bool)
        self.last=np.zeros(self.n_agents)
    def act(self,obs,mask,training=False,epsilon=0.):
        observation=np.asarray(obs,dtype=float);available=np.asarray(mask)
        if observation.shape!=(self.n_agents,3) or not np.isfinite(observation).all():
            raise ValueError('Rule requires finite normalized N x 3 observations')
        if available.shape!=(self.n_agents,len(ACTIONS)) or not np.isin(available,[0,1]).all():
            raise ValueError('Rule requires a boolean N x 16 action mask')
        available=available.astype(bool)
        if not available.any(axis=1).all():
            raise ValueError('Every rule-controlled cell must allow an action')
        state=observation*[.5,1.,11.]+[.5,3.5,308.]
        soc,v=state[:,0],state[:,1]
        if self.method=='soc_rule':
            current=np.clip(self.config.get('rule_base_A',3.)+self.config.get('rule_gain_A_per_soc',20.)*(soc.max()-soc),0,7.5)
        else:
            cc=float(self.config.get('cc_current_A',7.5))
            reference=4.2-float(self.config.get('voltage_margin_V',.03))
            self.cv|=v>=reference-.01
            # Incremental voltage feedback is an explicitly sampled CV loop.
            # The shared screen may further modify it, and those actions are logged.
            cv_current=np.clip(self.last+self.config.get('cv_gain_A_per_V',20.)*(reference-v),0,cc)
            current=np.where(self.cv,cv_current,cc)
        target=self.config['rule_terminal_soc_target']
        taper=np.maximum(self.config['rule_terminal_min_request_A'],
                         self.config['rule_terminal_taper_A_per_soc']*(target-soc))
        current=np.minimum(current,taper)
        current[soc>=target]=0.
        caps=np.max(np.where(available,ACTIONS,-np.inf),axis=1)
        current=np.minimum(current,caps)
        if self.action_kind=='discrete':
            current=np.floor(current/.5+1e-10)*.5
            # Honor arbitrary masks, not only the contiguous masks used by Pack.
            current=np.asarray([ACTIONS[row & (ACTIONS<=command+1e-10)].max()
                                if np.any(row & (ACTIONS<=command+1e-10)) else ACTIONS[row].min()
                                for row,command in zip(available,current)])
        return current,{'cv_stage':self.cv.tolist(),
                        'controller':('SOC-spread feedback with terminal SOC taper' if self.method=='soc_rule'
                                      else 'sampled CC-CV voltage feedback with terminal SOC taper; not ideal analog CV'),
                        'terminal_soc_target':float(target),
                        'terminal_taper_A_per_soc':float(self.config['rule_terminal_taper_A_per_soc']),
                        'terminal_min_request_A':float(self.config['rule_terminal_min_request_A']),
                        'taper_uses_fitted_or_true_capacity':False,
                        'global_completion_rule':'environment stops at min_SOC>=0.90 and std_SOC<=0.02'}
    def set_executed(self,current):
        actual=np.asarray(current,dtype=float)
        if actual.shape!=(self.n_agents,) or not np.isfinite(actual).all() or np.any(actual<0) or np.any(actual>7.5+1e-8):
            raise ValueError('Executed rule currents must be finite N-vector in [0,7.5] A')
        self.last=actual.copy()
    def learn(self,episodes):
        raise RuntimeError('Rule policy has no trainable parameters')
    def sync(self):
        pass
    def parameter_count(self):
        return 0

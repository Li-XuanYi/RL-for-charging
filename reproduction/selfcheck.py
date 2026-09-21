"""Small numerical checks; not evidence that training has converged."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from battery import Cell, INITIAL, DT, ACTIONS, Pack
from learner import Learner


def main():
    torch.set_num_threads(2)
    cell = Cell(INITIAL, .3)
    t0 = float(cell.sol.t[-1])
    accepted, _, vmax, tmax = cell.step(4.)
    assert 0 <= accepted <= 4 and .3 < cell.soc < .4
    assert abs(cell.sol.t[-1]-t0-DT)<1e-3
    assert vmax<=4.2+1e-5 and tmax<=309+1e-5
    charged = cell.soc
    cell.step(-1.)
    assert cell.soc < charged, "Current sign must represent charge versus discharge correctly"
    pack = Pack([INITIAL]*3,[.1,.2,.3])
    learner = Learner(7)
    episode = dict(obs=[pack.obs()],masks=[pack.mask()],actions=[],rewards=[],done=[])
    for _ in range(2):
        choice=learner.choose(pack.obs(),pack.mask(),.5)
        reward,done,stats=pack.step(ACTIONS[choice])
        episode['actions'].append(choice)
        episode['rewards'].append(reward)
        episode['done'].append(done)
        episode['obs'].append(pack.obs())
        episode['masks'].append(pack.mask())
        assert (stats[:,0]>=0).all(), "MBA-RL cannot discharge"
    assert not np.array_equal(episode['obs'][-1],episode['obs'][-2]), "Final observation must reflect executed action"
    short = {k:v[:1 if k in ('actions','rewards','done') else 2] for k,v in episode.items()}
    short['done']=[True]
    before = [p.detach().clone() for p in learner.target_agent.parameters()]
    value = learner.train([episode,short])
    assert np.isfinite(value)
    assert all(torch.equal(x,y) for x,y in zip(before,learner.target_agent.parameters())), "Targets cannot synchronize every gradient step"
    learner.sync()
    assert all(torch.equal(x,y) for x,y in zip(learner.agent.parameters(),learner.target_agent.parameters()))
    with tempfile.TemporaryDirectory() as d:
        checkpoint=Path(d)/'model.pt'
        learner.save(checkpoint,{'checked':True})
        second=Learner(9)
        assert second.load(checkpoint)['checked']
        assert all(torch.equal(x,y) for x,y in zip(learner.agent.parameters(),second.agent.parameters()))
    print('PASS: current sign, 90-second integration, safety bounds, nonnegative MBA actions, true next observation, padded TD update, target sync and checkpoint reload.')


if __name__=='__main__':
    main()

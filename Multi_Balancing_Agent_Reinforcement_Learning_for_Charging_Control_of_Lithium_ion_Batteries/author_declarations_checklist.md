# Author Declarations Checklist

The academic-paper revision protocol requires the following declarations, but the available manuscript files do not contain enough verified author information to draft them safely. Authors should complete and approve each item before submission.

- [ ] Data Availability Statement: identify the repository/DOI or confirm the conditions for access to raw cycling data, identified parameters, trained checkpoints, and evaluation traces.
- [ ] Code Availability Statement: identify the exact release/commit that reproduces the figures after resolving M2 and M3.
- [ ] Ethics Declaration: confirm that the work involves no human participants, animals, or sensitive personal data, or provide the applicable approval details.
- [ ] Author Contributions (CRediT): assign roles for Hejiakang Cao, Bing-Chuan Wang, Biao Luo, and Zhongmei Li; do not infer roles from author order.
- [ ] Conflict of Interest Statement: obtain an explicit declaration from every author.
- [ ] Funding Acknowledgment: provide grant names and numbers, or explicitly confirm that no external funding supported the work.
- [ ] AI-use Disclosure: disclose the use of an AI writing assistant for language editing, consistency checks, citation-metadata checking, and revision organization, following the target venue's policy.

## Experimental facts requiring author confirmation

- [ ] Confirm that each Samsung 40T cell was measured on an independently controlled NEWARE channel and that the cells were not directly hard-paralleled during independent-current tests.
- [ ] Confirm that “three independent identification runs were averaged” accurately describes the replicate procedure.
- [ ] Recover and approve the three distinct PSO-identified parameter dictionaries (Battery 1–3), including units, bounds, source run, and file hash; otherwise approve narrowing the paper to a shared-parameter Chen2020 simulation.
- [ ] Confirm that SOC-consistent PyBaMM initialization is the intended confirmatory protocol and that legacy mixed voltage/SOC initialization is used only for provenance diagnosis.
- [ ] Confirm the exact action grid, episode count, and target-network update interval that generated every published figure.
- [ ] Confirm whether Table II reports the result-generating configuration or only the intended configuration.

No declaration above has been inserted into the manuscript because doing so without author confirmation could create a false submission statement.

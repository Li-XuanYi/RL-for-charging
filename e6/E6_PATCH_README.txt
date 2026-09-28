E6 syntax repair, 2026-09-28

Copy the included files to the same relative paths in the Linux project root.
Do not modify existing E3/E4/E5 result folders or their lineage.json files.
This bridge accepts only the pinned old E6 and study_core hashes; every other source hash remains strictly checked.
Run a fresh E6 invocation (not --resume on an old failed folder):
  bash run_experiment_6_gpu_parallel_2500_5seeds_90s.sh
The GPU launcher sets up the environment and imports project paths before E6.
Import-only check after activating that environment from revision_gpu:
  python -c "import paths, experiment3, experiment4, experiment5, experiment6; print('IMPORT_OK')"
No E3-E5 training is required when all existing upstream hashes match the pinned manifest.
E5 status/lineage could not be checked on the authoring machine; the E6 gate checks it on the target machine.

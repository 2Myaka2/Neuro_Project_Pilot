# GraphSWARM v1 — stage 1 report

Prepared on 2026-10-05 (Europe/Moscow). Stage 1 infrastructure is complete.
No GraphSWARM model, new neural-network training, hyperparameter search, or
protected evaluation was performed. Stage 2 has not started.

## Repository and commit scope

Repository: `2Myaka2/Neuro_Project_Pilot`. Branch: **`swarm-advanced`**.
Audited starting/current commit before the single implementation commit:
`c9261bc9073fcdb92ff6cba36b514caf7e1a6902`.
At the initial check, HEAD, local main, origin/main, and remote main all matched
this commit, and the working tree was clean. The implementation commit is the
single commit containing this report; retrieve its exact hash with:

```bash
git log -1 --format=%H -- experiments/graph_swarm_v1/reports/stage1_report.md
```

The final response records that hash. This report records the audited parent
instead of attempting to embed its own containing commit hash.

Modified: `pyproject.toml` (discover `src/neuropp`, declare NumPy and Python >=3.10,
and expose the local CLI; the legacy README setting remains intact).

Created:

- `README.md` — experiment index, preserving the original bilingual READMEs.
- `src/neuropp/__init__.py`, `protocol.py`, `splits.py`, `graph.py`, `cache.py`,
  `data_access.py`, `provenance.py`, `local_artifacts.py`, `cli.py`.
- `tests/graph_swarm_v1/test_infrastructure.py`.
- `splits/graph_swarm_v1.json`.
- `experiments/graph_swarm_v1/README_eng.md`, `README_rus.md`, `protocol.json`,
  `config.yaml`, `AMENDMENTS.md`.
- `experiments/graph_swarm_v1/reports/stage1_report.md`, `feature_provenance.json`,
  `sequence_audit.json`, `graph_diagnostics.json`.

Three local graph cache entries were created under the ignored namespace
`data/graphs/graph_swarm_v1/`. Their locations appear in `graph_diagnostics.json`;
binary cache artifacts are not committed. No original feature cache was written.

## Validation commands and results

All commands ran from the repository root. CPU environment: `NeuroPP`,
Python 3.10, NumPy 2.2.6, PyTorch 2.5.1+cpu. No packages or data were downloaded.
The final test command was:

```bash
PYTHONPATH=src conda run -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -v
```

**50 passed / 0 failed / 0 skipped**; unittest runtime 0.465 seconds.
Synthetic tests cover the deterministic split recipe, ordering invariance,
counts, nonoverlap, source union, immutable membership, protected/unknown IDs,
batch preflight, source identity checks before label files open, separate label
storage, a manual fixed-k graph, symmetry, self-loop/duplicate exclusion,
short/singleton/empty proteins, NaN/Inf rejection, immutable position checks,
RBF centers, sequence separation, joint permutations including equal-distance
ties, rigid motions in float32/float64, alignment failures, residue mapping,
diagnostics, disconnected graphs, cache incompatibility/corruption, failed
atomic writes, missing artifacts, CLI protection, protocol constants, and
imports without downloads, notebooks, graph preparation, or training.

Synthetic labels in temporary per-protein fixtures were used only for access
tests. No real mutation labels were read. Numerical comparisons use float64
`atol=1e-8, rtol=1e-7` and float32 `atol=1e-5, rtol=1e-5`.

Real, metadata-only, and artifact audits ran successfully (exit code 0):

```bash
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli splits
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli audit --runtime --output experiments/graph_swarm_v1/reports/feature_provenance.json
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli sequence-audit --output experiments/graph_swarm_v1/reports/sequence_audit.json
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli check-real --protein-id 1AOY.pdb --protein-id 1E0L.pdb --protein-id 1I6C.pdb --output experiments/graph_swarm_v1/reports/graph_diagnostics.json
git diff --check
git diff --exit-code HEAD -- notebooks/01_baseline.ipynb notebooks/02_swarm_experiment.ipynb README_rus.md README_eng.md results
```

The split command was also rerun to validate the existing file, without changing
its bytes. The pinned original IDs were loaded from
`external/ThermoMPNN/dataset_splits/mega_splits.pkl`;
SHA256 `9e06230fe8febd8f07f8fab374f153328d4bdde20b78de15ead93b39e61ca1eb`.
The upstream artifact has extra cross-validation partitions and NumPy arrays;
only original train/val/test IDs are used, and the authenticated source is
normalized without changing any ID.

Final counts are **209 train / 31 validation / 30 head_holdout / 28 legacy_test**,
with no ID overlaps and an unchanged 298-protein union. The file SHA256 is:

```text
855312d024ef674274c7bc4cb40230156108fac947bf447b626844876410ec8d
```

WT PDB sequence metadata were checked for **all 298 proteins**. There were
**zero exact sequence duplicate groups crossing splits** and zero missing
structures. Protected WT metadata were read only for this sequence audit;
no protected feature extraction, graph construction, or evaluation occurred.
Proteins were not reassigned.

## Real graph checks and provenance

Feature provenance status: **passed**. The pinned ThermoMPNN commit is
`370f76ec62bd929f7425e311d8df04a0d094990f`; its tracked checkout was clean.
Both upstream checkpoint hashes match the owner-provided pins. Runtime
construction confirms ProteinMPNN parameters are frozen; the loaded core is
set to eval mode with gradients disabled. Extraction directly calls ProteinMPNN
and concatenates its two final hidden states and WT embedding into `[L,384]`.
Stability-head modules are excluded and guarded by hooks.

For the following train/validation proteins, existing frozen feature metadata,
PDB hash, encoded WT sequence, complete coordinates, and residue mapping were
checked. Fresh frozen features match the old cache at float32 tolerances.
**Stability-head calls: zero for every protein.** Graph cache reads validate
metadata, array checksums, and reconstructed graph compatibility.

| Protein | Split | L | Directed edges | Degree min / mean / max | Components | Edge distance min / max, Å | Reachable 1 / 2 / 4 hops |
|---|---|---:|---:|---|---:|---|---|
| 1AOY.pdb | train | 69 | 1314 | 16 / 19.0435 / 32 | 1 | 3.7899 / 14.3077 | 0.2801 / 0.7758 / 1.0000 |
| 1E0L.pdb | train | 37 | 724 | 16 / 19.5676 / 32 | 1 | 3.7873 / 18.4920 | 0.5435 / 0.9835 / 1.0000 |
| 1I6C.pdb | validation | 39 | 776 | 16 / 19.8974 / 32 | 1 | 3.7043 / 23.0036 | 0.5236 / 0.9703 / 1.0000 |

The full unrounded diagnostics, source-feature file hashes, and cache paths are
in [graph_diagnostics.json](graph_diagnostics.json). The 23 Å edge illustrates
that distances are not clipped at the final 20 Å RBF center. Degrees above 16
are expected after symmetric union. Complete four-hop reachability was recorded
without changing k, topology, or the scientific specification.

## Preservation and protected access

- **Protected labels were not accessed.** Neither head_holdout nor legacy_test
  mutation labels were loaded. No bulk MegaScale CSV was opened. No protected
  evaluator or unlock was implemented.
- **Legacy experiment files were not modified.** The two original notebooks,
  root bilingual READMEs, and all published files under `results/` match the
  starting commit. Notebook source was inspected without execution or saved
  outputs. `results/swarm_final/` was inventoried by filename only; no error or
  prediction tables were opened. The original READMEs were inspected as requested;
  their published performance was not used for architecture selection.
- Existing ThermoMPNN feature caches and old trained head weights were not
  written. Selected old feature dictionaries contain legacy baseline scores;
  the adapter reads only metadata/features/mask/encoded-sequence fields into
  the record and never uses those scores as inputs or checks them for selection.
- Main was not modified; no reset, force-push, merge, push, or stage 2 operation
  was performed.

## Blocked work, limitations, and owner commands

**No required stage 1 artifact is missing in this workspace.** Artifact-blocked
paths are implemented and tested synthetically. On another machine, provide
the exact pinned checkout, checkpoints, original split, WT PDBs, and compatible
existing frozen feature cache locally, then rerun the commands above. A missing
artifact returns `blocked / missing artifact`; it is never treated as passed.

Pending future work, outside this stage:

- GraphSWARM and comparison models, model dimensions and hidden iteration
  choices, training, and eventual protected evaluation require separate scope.
  No architecture or training hyperparameters have been selected here.
- Future labelled development requires owner-supplied isolated train/validation
  files for `load_label_directory`. Their layout is documented in the loader;
  no dataset label export or real labelled-loader run was performed in stage 1.
- Only three real graphs/features were audited; the remaining proteins have
  sequence metadata checks, not complete graph/feature audits. Graph invariance,
  edge cases, corruption, and access failure scenarios were tested synthetically.
- The access mechanism guards project APIs, not arbitrary raw filesystem reads;
  future code must route labelled loading through it.
- Exact sequence duplicates are audited; sequence similarity, homologous groups,
  and backbone pretraining contamination are not. `head_holdout` protects only
  the new heads and must not be called an independent benchmark for the system.
- The strict PDB adapter intentionally rejects multi-chain/multi-model, alternate
  location, nonstandard, incomplete, or ambiguous inputs instead of repairing them.
- Graph construction uses O(L²) pairwise distance memory; it is suitable for
  the small proteins in this experiment, not yet audited for large proteins.
- Four graph steps cover all nodes in the sampled proteins. This may limit
  a future locality interpretation, but does not change the protocol and does
  not establish physical propagation.
- The pinned upstream loader emits a PyTorch future warning about the default
  checkpoint decoder. It operates only on hash-verified local official weights;
  upstream source files were preserved. Runtime audit dependencies are supplied
  by `environment.yml`; the synthetic infrastructure requires NumPy only.

The owner can rerun the validation commands above. Optional offline editable
installation is documented in both experiment READMEs. No further artifact
provision or approval is needed to review this stage 1 commit. Proceeding to
stage 2 requires a new instruction.

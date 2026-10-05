# GraphSWARM v1 — Stage 1 and corrective Stage 1.1 report

Prepared and corrected on 2026-10-05 (Europe/Moscow). Stage 1 infrastructure
and the requested Stage 1.1 corrective audit are complete.
No GraphSWARM model, new neural-network training, hyperparameter search, or
protected evaluation was performed. Stage 2 has not started.

## Repository and commit scope

Repository: `2Myaka2/Neuro_Project_Pilot`. Branch: **`swarm-advanced`**.
Original Stage 1 commit: **`af3f190970ffd312d0c165de3194d2e69f17a382`**, whose
implementation parent was `c9261bc9073fcdb92ff6cba36b514caf7e1a6902`.
Before Stage 1.1 changes, the branch was confirmed as `swarm-advanced`, HEAD
matched the original Stage 1 commit exactly, and the working tree was clean.

Stage 1.1 is one corrective child commit of the original Stage 1 commit.
Retrieve its containing commit hash with:

```bash
git log -1 --format=%H -- experiments/graph_swarm_v1/reports/stage1_report.md
```

The final response records that hash. This report identifies the original
Stage 1 commit explicitly without embedding its own containing commit hash.
Stage 1.1 changes only split eligibility, tensor provenance, their tests, and
experiment documentation/reports. It adds `test_provenance.py`; the original
graph implementation, graph parameters, access policy, packaging, and YAML
configuration are unchanged. The three existing local graph cache entries were
validated and reused; no original feature cache or weights were written.

## Stage 1.1 correction and inspection evidence

Only the four permitted legacy documents were reviewed initially:
`notebooks/01_baseline.ipynb`, `notebooks/02_swarm_experiment.ipynb`,
`README_rus.md`, and `README_eng.md`. The first notebook identifies `1A32.pdb`
as original train and explicitly displays five experimental mutation labels,
predictions, individual errors, and MAE in cell 14 (zero-based cell index).
The second notebook displays aggregate metrics, without additional individual
original-training mutation labels/errors. The root READMEs repeat the 1A32
example; their named legacy-test examples are aggregate protein comparisons.
The authenticated original split independently confirms 1A32 train membership.
The confirmed previously explicitly inspected original-training set is exactly
**`{"1A32.pdb"}`**. No old prediction/error table was opened.

The original rule ranked all 239 original train IDs by the frozen SHA256 rule
and took the first 30. Stage 1.1 makes `1A32.pdb` ineligible for `head_holdout`
and keeps it in train. The other 238 IDs retain the exact prefix
`graph_swarm_v1:20261004:` and `(hash, protein_id)` sorting; the first 30 eligible
IDs become `head_holdout`. All remaining original train IDs become train.

**`1A32.pdb` left head_holdout; `2MA4.pdb` entered head_holdout.** Those are the
only membership changes. The generator implements this rule; the JSON was
regenerated from authenticated original IDs after checking its previous SHA256
and the exact set changes. Ordinary CLI validation still refuses overwriting a
different existing manifest. See [AMENDMENTS.md](../AMENDMENTS.md) for the
authorization, previous/revised rules, date, and reason.

No GraphSWARM model had been implemented or trained at correction time.
The correction was based solely on documented prior human inspection and
**was not based on GraphSWARM performance**. Graph protocol/cache compatibility
remains `graph_swarm_v1:1`; split eligibility is explicitly marked `stage1.1`.

## Validation commands and results

All commands ran from the repository root. CPU environment: `NeuroPP`,
Python 3.10, NumPy 2.2.6, PyTorch 2.5.1+cpu. No packages or data were downloaded.
The final test command was:

```bash
PYTHONPATH=src conda run -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -v
```

Original Stage 1: **50 passed / 0 failed / 0 skipped**.
Stage 1.1: **65 passed / 0 failed / 0 skipped**; final unittest runtime 0.560 seconds.
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
The 15 added tests cover 1A32 train membership across shuffled inputs,
deterministic replacement, exclusion enforcement even after hash recomputation,
and full tensor comparison including buffers, keys, shapes, dtypes, finiteness,
single-float32-ULP changes, altered frozen weights loaded from a synthetic
ThermoMPNN checkpoint, and CLI failure reports before feature loading.

Synthetic labels in temporary per-protein fixtures were used only for access
tests. No dataset mutation labels were loaded. The permitted saved 1A32 notebook
display was inspected as described above. Graph/feature numerical comparisons
use float64 `atol=1e-8, rtol=1e-7` and float32 `atol=1e-5, rtol=1e-5`.
Stored ProteinMPNN weights use **exact equality with no tolerance**.

All four audits were rerun for Stage 1.1 and passed (exit code 0):

```bash
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli splits
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli audit --runtime --output experiments/graph_swarm_v1/reports/feature_provenance.json
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli sequence-audit --output experiments/graph_swarm_v1/reports/sequence_audit.json
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli check-real --protein-id 1AOY.pdb --protein-id 1E0L.pdb --protein-id 1I6C.pdb --output experiments/graph_swarm_v1/reports/graph_diagnostics.json
git diff --check
git diff --exit-code --quiet af3f190970ffd312d0c165de3194d2e69f17a382 -- notebooks/01_baseline.ipynb notebooks/02_swarm_experiment.ipynb README_rus.md README_eng.md results
```

The split command was also rerun to validate the existing file, without changing
its bytes. The pinned original IDs were loaded from
`external/ThermoMPNN/dataset_splits/mega_splits.pkl`;
SHA256 `9e06230fe8febd8f07f8fab374f153328d4bdde20b78de15ead93b39e61ca1eb`.
The upstream artifact has extra cross-validation partitions and NumPy arrays;
only original train/val/test IDs are used, and the authenticated source is
normalized without changing any ID.

Final counts are **209 train / 31 validation / 30 head_holdout / 28 legacy_test**,
with no ID overlaps and an unchanged 298-protein union. Validation and legacy_test
lists, original-list hash, source hash, and selection prefix are unchanged.
Original Stage 1 split file SHA256:

```text
855312d024ef674274c7bc4cb40230156108fac947bf447b626844876410ec8d
```

**Revised Stage 1.1 split file SHA256:**

```text
310839d388e4430a817f849fb9c3f08ca563bc074706bb9b9facffbcb1aeb796
```

WT PDB sequence metadata were checked for **all 298 proteins**. There were
**zero exact sequence duplicate groups crossing splits** and zero missing
structures. Protected WT metadata were read only for this sequence audit;
no protected feature extraction, graph construction, or evaluation occurred.
The sequence audit itself did not reassign proteins; the only changes are the
authorized eligibility correction described above. Its JSON is byte-identical
to the original report because the result remains 298 checked / zero duplicates.

## Real graph checks and provenance

Feature provenance status: **passed**. The pinned ThermoMPNN commit is
`370f76ec62bd929f7425e311d8df04a0d094990f`; its tracked checkout was clean.
Both upstream checkpoint hashes match the owner-provided pins. Stage 1.1 directly
compares the **full ProteinMPNN state_dict after loading the pinned ThermoMPNN
checkpoint** against `model_state_dict` in the pinned original
`external/ThermoMPNN/vanilla_model_weights/v_48_020.pt`.
The original ProteinMPNN file SHA256 is
`c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`.

```yaml
proteinmpnn_tensor_match:
  passed: true
  compared_tensors: 118
  exact_match: true
  max_abs_difference: 0.0
  key_set_equal: true
  shapes_equal: true
  dtypes_equal: true
```

Every parameter and buffer is covered. All values are finite; key sets, shapes,
and dtypes match exactly. `torch.equal` verifies values without tolerance or
fallback. A failed comparison blocks provenance and graph checks and retains
the maximum discrepancy and tensor diagnostics in the CLI report. The evidence
appears at `checks.proteinmpnn_tensor_match` in
[feature_provenance.json](feature_provenance.json) and in the provenance embedded
in [graph_diagnostics.json](graph_diagnostics.json). The feature signature is
unchanged: `55c5ee9e7e9c264ce3a110bb22a64ba3872726d976d2905f345635b4a83bb991`.

Runtime construction also confirms ProteinMPNN parameters are frozen; the loaded core is
set to eval mode with gradients disabled. Extraction directly calls ProteinMPNN
and concatenates its two final hidden states and WT embedding into `[L,384]`.
Stability-head modules are excluded and guarded by hooks.

The same three proteins remain development-accessible under the revised split.
For these train/validation proteins, existing frozen feature metadata,
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
  mutation labels were loaded. The permitted saved 1A32 notebook display was
  inspected to document prior human exposure; 1A32 now belongs to train.
  No bulk MegaScale CSV or old result prediction/error table was opened.
  No protected evaluator or unlock was implemented.
- **Legacy experiment files were not modified.** The two original notebooks,
  root bilingual READMEs, and all published files under `results/` match the
  original Stage 1 commit, verified by the quiet Git diff above. Notebook source
  and saved displays in the four permitted documents were inspected without
  executing or modifying either notebook. No file inside `results/swarm_final/`
  was opened for this correction. Published performance was not used to change
  eligibility or model design.
- Existing ThermoMPNN feature caches and old trained head weights were not
  written. Selected old feature dictionaries contain legacy baseline scores;
  the adapter reads only metadata/features/mask/encoded-sequence fields into
  the record and never uses those scores as inputs or checks them for selection.
- Main was not modified; no reset, force-push, merge, push, or stage 2 operation
  was performed.
- **No GraphSWARM implementation, training, or evaluation occurred.** No
  GraphSwarmLayer, GraphStatic, MutationSelf, training loop, model notebook,
  bootstrap, or protected evaluator was added. No weights were downloaded.

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
  by `environment.yml`. Existing infrastructure tests require NumPy; the new
  synthetic tensor tests also require PyTorch and explicitly skip if absent.
  PyTorch was available here, so all 65 tests passed with no skips.

The owner can rerun the validation commands above. Optional offline editable
installation is documented in both experiment READMEs. No further artifact
provision or approval is needed to review this stage 1 commit. Proceeding to
stage 2 requires a new instruction.

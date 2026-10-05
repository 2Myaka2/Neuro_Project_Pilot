# Protocol amendments

Protocol: `graph_swarm_v1:1`. Specification date: 2026-10-04.
Stage 1 infrastructure prepared on 2026-10-05.

## Stage 1.1 correction — 2026-10-05 (Europe/Moscow)

Original Stage 1 commit: `af3f190970ffd312d0c165de3194d2e69f17a382`.
Authorization: the owner's corrective Stage 1.1 request, before Stage 2.

**Previous rule:** rank all 239 original-training IDs by
`SHA256(UTF8("graph_swarm_v1:20261004:" + protein_id))`, sort by `(hash, protein_id)`,
and put the first 30 in `head_holdout`. This included `1A32.pdb`.

**Revised eligibility rule:** `1A32.pdb` is ineligible for `head_holdout` and
must remain in train. Rank the other 238 original-training IDs using the same
frozen SHA256 prefix and sorting rule; take the first 30 eligible proteins.
All remaining original-training proteins, including `1A32.pdb`, become train.
Counts remain 209 train / 31 validation / 30 head_holdout / 28 legacy_test.
`1A32.pdb` leaves `head_holdout`; `2MA4.pdb` enters it. Validation, legacy_test,
and the 298-protein union are unchanged.

**Reason:** prior explicit human inspection of individual experimental mutation
labels and errors for `1A32.pdb`. The permitted review covered only
`notebooks/01_baseline.ipynb`, `notebooks/02_swarm_experiment.ipynb`,
`README_rus.md`, and `README_eng.md`. The first notebook explicitly displays
five mutation labels, predictions, errors, and their MAE. The second notebook
and root READMEs introduce no additional original-training protein with
individually displayed mutation labels/errors. The confirmed set is exactly
`{"1A32.pdb"}`. No old prediction/error tables were opened.

**Timing:** no GraphSWARM model had been implemented or trained when this
correction was made. The correction was not based on GraphSWARM performance;
no GraphSWARM training or evaluation occurred. No protected labels were loaded.
The authorized inspection of the saved 1A32 notebook display is recorded above.

The split generator encodes the exclusion and records it in the manifest.
The manifest is explicitly regenerated from the authenticated original IDs
after checking the previous file hash and the exact two membership changes.
The ordinary CLI continues to refuse overwriting a different existing split.
Affected split artifacts: generator/constants, split tests, protocol.json,
both experiment READMEs, split manifest, this amendment, and Stage 1 report.

The runtime provenance correction compares every ProteinMPNN parameter and
buffer **after** the pinned ThermoMPNN checkpoint is loaded, against
`model_state_dict` in the pinned original `vanilla_model_weights/v_48_020.pt`.
Keys, shapes, and dtypes must match, and finite values must satisfy `torch.equal`.
There is no tolerance or fallback; failure preserves tensor diagnostics and
blocks provenance and real graph checks. Synthetic altered-checkpoint and
single-ULP tests cover this requirement. Feature/graph cache compatibility has
not changed: `graph_swarm_v1:1` remains the graph protocol version, and
`stage1.1` identifies the amended split eligibility rule. The selection prefix
remains `graph_swarm_v1:20261004:`.

## Existing engineering conventions

Engineering conventions make the supplied specification explicit:

- `sequence` remains in canonical WT order when array rows are permuted.
  `seq_pos` is an immutable, zero-based WT position; a full protein contains
  each position exactly once. Mutation positions use the same convention.
- PDB residue UIDs are `chain:residue_number:insertion_code`; alignment is checked
  against both PDB residue letters and decoded frozen-feature WT letters.
- Hop reachability is averaged over starting nodes, excludes the starting node,
  and divides by `L-1`. It is defined as zero for a singleton.
- Cache contents contain only graph arrays and integrity/provenance metadata.
  Hashing frozen features additionally prevents incompatible feature reuse.
- The upstream split pickle contains NumPy arrays. Its general decoder is used
  only after matching the exact owner-pinned SHA256; unpinned pickles cannot
  resolve Python globals.
- The YAML file records the frozen specification and paths. CLI flags specify
  local artifact locations; scientific graph parameters cannot be overridden.

Future scientific changes require an explicit amendment explaining the reason,
date, prior and replacement values, affected artifacts, and authorization.
Version the protocol and cache namespace when compatibility changes. Do not
silently replace the split, unlock protected labels, or change k/RBF features.

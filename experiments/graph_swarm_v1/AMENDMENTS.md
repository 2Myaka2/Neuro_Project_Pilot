# Protocol amendments

Protocol: `graph_swarm_v1:1`. Specification date: 2026-10-04.
Stage 1 infrastructure prepared on 2026-10-05.

No scientific amendments have been made. The selection prefix remains
`graph_swarm_v1:20261004:` regardless of the implementation date.

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

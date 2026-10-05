# NeuroPP experiments

This repository studies protein stability prediction from MegaScale, WT structures,
and frozen ThermoMPNN / ProteinMPNN representations.

- **Experiment 1 — ThermoMPNN + global SWARM:** completed pilot experiment.
  Its original [Russian](README_rus.md) and [English](README_eng.md) documentation,
  notebooks, and published `results/swarm_final/` outputs are preserved.
- **Experiment 2 — Mutation-Conditioned GraphSWARM v1:** in implementation.
  Stage 1 prepares the protocol, split, access protection, protein graphs, cache,
  provenance audit, and tests. Models and training are future work.
  See the [English](experiments/graph_swarm_v1/README_eng.md) or
  [Russian](experiments/graph_swarm_v1/README_rus.md) specification and
  [stage 1 report](experiments/graph_swarm_v1/reports/stage1_report.md).

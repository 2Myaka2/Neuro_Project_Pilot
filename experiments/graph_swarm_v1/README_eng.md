# Mutation-Conditioned GraphSWARM v1

Stages 1/1.1 prepare infrastructure; Stage 2 implements the frozen model
architectures and synthetic model tests. Real-data training, hyperparameter
search, bootstrap analysis, and the protected evaluator are not implemented. Experiment 1
is completed and its notebooks, documentation, features, weights, and published
outputs are preserved. The machine-readable contract is [protocol.json](protocol.json);
[config.yaml](config.yaml) records fixed parameters and artifact locations.

## Scientific question

Does repeatedly recomputing local residue-to-residue messages from updated
mutation-conditioned hidden states improve ΔΔG prediction compared with using
a structural context that is computed once and then kept fixed?

The future primary comparison is `graph_static` versus `graph_swarm`. The
primary effect is
`protein_macro_MAE(graph_static) - protein_macro_MAE(graph_swarm)`; positive
values favor GraphSWARM. Protein macro MAE averages each protein's mutation MAE
with equal weight per protein. Stage 1 does not estimate this effect.

GraphSWARM is a **working project name** for a recurrent graph neural network
inspired by SWARM mappings. Hidden iterations are **not physical time**. The
model does **not simulate molecular dynamics**. Improved MAE alone does not
prove physical propagation of energy, force, or conformational waves.

## Data, provenance, and access

Use the same MegaScale dataset, WT structures, and frozen ThermoMPNN /
ProteinMPNN features as experiment 1: `[L,384]`, comprising two decoder hidden
states and the WT embedding, each with 128 components. Preserve the upstream
`hidden[:2]` order, followed by the WT embedding. These contextual features are
extracted **before** LightAttention and the final stability prediction head;
`light_attention`, `both_out`, and `ddg_out` are excluded from feature extraction.
Previous predictions, errors, old trained head weights, and legacy test
performance are forbidden model inputs. Positive ΔΔG means destabilization,
using the existing `-ddG_ML` convention in kcal/mol.

`neuropp.provenance` checks the pinned commit and hashes recorded in the protocol,
the checkpoint, runtime ProteinMPNN freezing, and the pre-head extraction path.
After loading the pinned ThermoMPNN checkpoint, it compares the full loaded
ProteinMPNN `state_dict` with `model_state_dict` from the pinned original
`vanilla_model_weights/v_48_020.pt`: identical keys (including buffers), shapes,
dtypes, and finite tensor values using `torch.equal`, with no tolerance. Any
mismatch stops the audit and reports the maximum absolute difference; provenance
can pass only when this comparison passes. Freezing alone is insufficient.
The real-data audit compares this path with existing frozen features while
hooks explicitly forbid stability-head calls. Missing artifacts produce
`blocked / missing artifact`; an artifact-only audit stays `pending` until
runtime checks. Nothing downloads automatically. The local check imports the
upstream model definitions directly and never imports its training entry point.

The original split is 239 train / 31 validation / 28 test. The old test is now
`legacy_test`: it was inspected previously and is not a new independent test.
The new split is **209 train / 31 validation / 30 head_holdout / 28 legacy_test**.
`head_holdout` protects the **new heads only**; it is not a fully independent
external benchmark for the entire system or its pretrained backbone.

Stage 1.1 excludes **`1A32.pdb`** from `head_holdout` eligibility and keeps it in
train because its individual mutation labels and errors were explicitly displayed
in `notebooks/01_baseline.ipynb`. The four allowed legacy documents confirmed
that it is the only original-training protein with such displays. This correction
was authorized before GraphSWARM implementation or training and uses no GraphSWARM
performance. **`2MA4.pdb`** replaces `1A32.pdb` in `head_holdout`.

For each of the other 238 unaltered original train protein IDs, hash UTF-8
`"graph_swarm_v1:20261004:" + protein_id` with SHA256. Sort by `(hash, protein_id)`,
choose the first 30 eligible proteins for `head_holdout`; all remaining original
train IDs, including `1A32.pdb`, become train. Store all lists lexicographically.
[The split manifest](../../splits/graph_swarm_v1.json) contains source hashes,
counts, the recipe, and hashes of the original and generated lists. Existing
splits are validated, and different splits are never silently overwritten.
Eligibility uses documented prior inspection; hash ranking uses no label values.
Exact cross-split WT sequence duplicates are reported
without reassignment; sequence similarity and homology are not assessed here.

All future labelled development loading must use `DevelopmentDataAccess`.
It verifies actual protein ID membership and preflights the entire batch before
calling a per-protein reader. Both protected splits raise `ProtectedLabelAccess`,
including when a caller claims `train`. Unknown IDs raise `UnknownProteinID`.
The concrete `load_label_directory` additionally checks metadata-only source
identities before opening any label file, validates mutation WT alignment, and
returns labels separately as `MutationLabels`. It accepts owner-supplied isolated
train/validation files, never a bulk CSV containing protected labels.
`metadata_only` has a separate `ProteinMetadata` contract. No unlock exists.
The API guard does not prevent someone deliberately bypassing it with raw file I/O.

## Protein records and graphs

`ProteinRecord` contains `protein_id`, canonical WT `sequence`, `features [L,384]`,
`ca_coordinates [L,3]`, `seq_pos [L]`, `residue_uid [L]`, `edge_index [2,E]`, and
`edge_attr [E,17]`. PDB and frozen-feature amino acid arrays provide alignment
evidence. Coordinates are in Å. Sequence stays in canonical order; array rows
may be jointly permuted while retaining immutable zero-based `seq_pos` and UIDs.
Mutation queries carry `mutation_position`, `wt_index`, and `mut_index`; labels
never enter graph construction or model-input preprocessing.

The graph uses Euclidean C-alpha distances. Each node selects
`min(16,L-1)` nearest other residues, with equal-distance ties resolved by
original WT sequence position. Take the symmetric union and store both
orientations: `edge_index[0]=source j`, `edge_index[1]=destination i`. No self-loops,
duplicate directed edges, cutoff edges, or extra sequence-neighbor edges exist.
Degrees may exceed 16 and connectivity is not guaranteed. A singleton has
`[2,0]` edges and `[0,17]` attributes; an empty protein raises an error.

The 17 features are 16 Gaussian RBFs with centers uniformly spaced from 2 to
20 Å inclusive and σ=1.2 Å, followed by
`log(1+abs(seq_pos_i-seq_pos_j))/log(1+L)`. Distances are never clipped.
Validation checks lengths, finite values, full WT alignment, and unambiguous
residue mapping. The local adapter requires a single model and chain, standard
residues, and complete N/CA/C/O backbone atoms. Alternate locations, missing
residues, inconsistent features, and ambiguous mappings fail explicitly.
No truncation, padding, repair, or silent index shifts are performed.

Diagnostics report degree min/mean/max, components, edge-distance bounds, and
the mean reachable fraction of other residues within 1/2/4 hops. The fraction
excludes the start residue and is zero for a singleton. Small proteins may
already be fully covered within four graph steps.

`GraphCache` uses `data/graphs/graph_swarm_v1/`, independently of the read-only
`data/features/thermompnn/`. Keys include protein ID, sequence, coordinates,
residue mapping, feature hashes, graph parameters, and protocol version. Entries
contain graph arrays and metadata only. Writes use a temporary file and atomic
replacement; reads verify metadata, checksums, and the graph contract. Corruption
raises an explicit error. Local caches remain ignored by Git.

## Local commands

Run from the repository root using the existing CPU `NeuroPP` environment.
These commands need no installation or network access:

```bash
PYTHONPATH=src conda run -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -v
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli splits
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli audit --runtime
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli sequence-audit
PYTHONPATH=src conda run -n NeuroPP python -m neuropp.cli check-real --protein-id 1AOY.pdb --protein-id 1E0L.pdb --protein-id 1I6C.pdb
```

The first command uses synthetic CPU data only. The last command uses only the
listed train/validation proteins and their existing files. `sequence-audit`
reads WT PDB metadata across all splits, including protected metadata, without
labels. Audit commands accept `--output path.json`; missing or incompatible
artifacts produce a blocked report and exit code 2. `check-real` never writes
the original feature cache. Optional local packaging:
`conda run -n NeuroPP python -m pip install --no-index --no-deps --no-build-isolation -e .`.
The installed CLI is `graph-swarm-v1` with the same subcommands.

For missing files, the owner must provide the pinned local checkout/checkpoints,
original split, WT structures, and compatible frozen feature cache at the
documented paths (or use location flags). Stage 1 stops at that boundary; it
does not download or create replacement datasets or weights. See
[AMENDMENTS.md](AMENDMENTS.md) and the [stage 1 report](reports/stage1_report.md)
for audited infrastructure evidence. The [Stage 2 report](reports/stage2_report.md)
records synthetic model evidence; real-data training requires a later stage.

## Stage 2 frozen model specification

The main variants are `mutation_self`, `graph_static`, and `graph_swarm` with
T=4. The separately instantiated secondary control `graph_swarm_t1` has T=1.
The primary future comparison remains `graph_static` versus `graph_swarm`:
identical registered modules, parameter tensors, and `state_dict` structure;
the only difference is message refresh after hidden-state updates.

For alphabet `ACDEFGHIKLMNPQRSTVWY`, indices 0..19 encode
`q = concat(one_hot(WT), one_hot(MUT), one_hot(MUT)-one_hot(WT))`, shape `[B,60]`.
WT and MUT must differ. No amino-acid properties, learned embeddings or labels
enter the head. Frozen `[L,384]` features are projected by a biased 384→64 linear
layer followed by tanh. The biased 60→64→64 mutation MLP has SiLU after its first
linear only. Its output is added exclusively to the row whose immutable
`seq_pos` equals the mutation position, producing `[B,L,64]` initial states.

For each directed edge j→i, concatenate receiver hidden state, sender hidden
state and the unchanged 17 edge attributes, in that order (145 dimensions).
The biased message MLP is 145→64→64; attention is 145→32→1, with bias on the
first linear and **no bias** on the final scalar linear. Both use SiLU only
after their first linear. Stable softmax normalizes separately over incoming
edges of each destination and each mutation query. Weighted messages sum at
the destination; nodes with no incoming edges receive exactly zero. The
implementation uses native PyTorch scatter operations without graph libraries.

All messages at step t are computed from Hᵗ before a single synchronous update
with `GRUCell(64,64,bias=True)`. One cell is shared across residues, steps and
queries. There is no communication across mutation queries, dropout, BatchNorm,
LayerNorm, extra residual, or global population vector inside recurrence.
`graph_swarm` refreshes both values and attention every step. `graph_static`
computes mutation-conditioned m⁰ once **after injection**, then reuses that
exact tensor at all four updates; it is recomputed on every forward and never
detached or constructed under `no_grad`.

`mutation_self` instead concatenates `(h_i,h_i,zeros(17))` and applies the same
message MLP every step, without attention. This means **no cross-node exchange
inside recurrent dynamics**. Other residues still influence the prediction
through the final global mean; this is not an isolated single-residue predictor.
Its attention module is registered but inactive. `graph_swarm_t1` uses the same
graph modules for one update. A T=1 diagnostic override of the primary variants
allows prediction and parameter-gradient equivalence tests, without adding a
new architecture or changing the frozen main T=4 comparison.

The final readout concatenates `(h_mutation_row, mean_i(h_i), q)` (188 dimensions)
and applies a biased 188→64→1 MLP with SiLU after its first linear only. The
output is one scalar predicted ΔΔG per query, shape `[B]`.

| Variant | Steps | Registered trainable parameters | Effectively used parameters |
|---|---:|---:|---:|
| mutation_self | 4 | 88,033 | 83,329 |
| graph_static | 4 | 88,033 | 88,033 |
| graph_swarm | 4 | 88,033 | 88,033 |
| graph_swarm_t1 | 1 | 88,033 | 88,033 |

The inactive attention accounts for 4,704 parameters. Effective counts describe
architectural participation in an ordinary injected forward, not nonzero
gradients of every scalar for every graph (e.g. singleton neighborhoods).

## Model API and diagnostics

Import `MutationGraphModel` from `neuropp.models`. PyTorch ≥2.5 is needed for
models and is already present in the CPU `NeuroPP` environment; the optional
package extra is `models`. Importing the base infrastructure still avoids
importing PyTorch. No package installation or download is needed here.

```python
model = MutationGraphModel("graph_swarm")  # T=4
prediction = model(features, seq_pos, edge_index, edge_attr,
                   mutation_positions, wt_indices, mut_indices)
prediction, diagnostics = model(
    features, seq_pos, edge_index, edge_attr,
    mutation_positions, wt_indices, mut_indices,
    return_diagnostics=True,
)
reference_prediction, reference = model(
    features, seq_pos, edge_index, edge_attr,
    mutation_positions, wt_indices, mut_indices,
    return_diagnostics=True, inject_mutation=False,
)
```

Inputs have shapes `[L,384]`, `[L]`, `[2,E]`, `[E,17]`, `[B]`, `[B]`, `[B]`.
L and B must be positive. Use float32/float64 matching the model dtype/device
and integer index tensors on that device; conversions are explicit. Forward
checks shapes, finiteness, full immutable position mapping, query index ranges,
WT≠MUT, and graph endpoint validity, self-loop and duplicate exclusion. It does
not reconstruct topology or infer WT sequence from contextual features:
`ProteinRecord.validate()` and `MutationQuery.validate(record)` retain Stage 1
geometric and WT-alignment responsibilities upstream. Synthetic chain tests
exercise the head with artificial sparse graphs; the real graph contract stays k=16.
Forward accepts no labels or arbitrary target arguments.

Diagnostics are off by default and return H⁰..Hᵀ, per-step messages and graph
attention, q, and mapped mutation rows. They preserve predictions, parameters,
and gradients and return live, attached tensors. Do not mutate them; release
them when finished. Static diagnostics repeat the same message/attention tensor
by identity; self attention entries are `None`.

The matched reference uses the same features, graph, weights, q and model mode,
omitting only local ψ(q) injection. A later response measure may use
`delta_i^t = ||h_i,injected^t - h_i,reference^t||₂`. This infrastructure is not
mechanistic evidence and is not WT molecular dynamics. Readout q remains
present in both passes; compare hidden trajectories, not raw Hᵗ−H⁰.

`model.checkpoint_configuration()` explicitly records variant, actual steps,
graph protocol version and the full architecture specification. Save it beside
`state_dict`; a future Stage 3 loader must compare both. Tensor keys alone cannot
distinguish static from refreshed messages or T=1 from T=4. Stage 2 implements
no training/checkpoint pipeline or protected evaluator.

## Interpretation and allowed Stage 2 validation

Stage 1's three sampled real graphs (`1AOY.pdb`, `1E0L.pdb`, `1I6C.pdb`) reached
**all nodes within four hops** (mean fraction of other residues = 1.0000).
Therefore T=4 does not guarantee local-only information. Four updates are not
four physical interaction shells, physical time steps, or molecular signal
propagation. The question is whether recomputing messages from evolving
mutation-conditioned states helps relative to fixed mutation-conditioned messages.
Synthetic chain bounds concern the **additional head response at fixed frozen
features**, not ProteinMPNN's receptive field.

Run only Stage 1/1.1 and Stage 2 synthetic tests during this stage:

```bash
PYTHONPATH=src conda run --no-capture-output -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -v
```

The tiny learning test optimizes four prescribed synthetic targets for 60 CPU
updates as an engineering check. It uses no MegaScale labels, validation proteins
or performance-based tuning. Do not run the earlier real-artifact audit commands
as part of Stage 2. Real training, validation MAE, bootstrap/protected evaluation,
and `03_mutation_graph_swarm.ipynb` are deferred. Synthetic success does not
establish real-data accuracy or scientific benefit.

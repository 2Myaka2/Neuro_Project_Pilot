# GraphSWARM v1 — Stage 2 model and synthetic-test report

Completed on 2026-10-05 (Europe/Moscow) in repository
`2Myaka2/Neuro_Project_Pilot`, branch **`swarm-advanced`**.
Stage 2 implements models and synthetic/model-level tests only. It makes no
real-data performance or mechanistic claim and stops before real training.

## Starting point and review scope

Required and verified parent: **`646b00819f435e6143a8e42af23de840c681d2e3`**.
Before edits, the current branch matched `swarm-advanced`, HEAD matched that
exact commit, the working tree was clean, and all **65 Stage 1/1.1 tests passed**
(0 failed, 0 skipped; unittest runtime 0.558 s). No prerequisite failed.

One Stage 2 commit contains the following eight files:

| File | Change |
|---|---|
| `src/neuropp/models.py` | New shared model, encoding, configuration, diagnostics, native scatter aggregation and parameter-count utility |
| `tests/graph_swarm_v1/test_models.py` | New 44 synthetic/model-level tests, including a fixed tiny learning check |
| `pyproject.toml` | Optional `models` extra requiring PyTorch ≥2.5; base NumPy infrastructure unchanged |
| `experiments/graph_swarm_v1/protocol.json` | Separate frozen model section and Stage 2 status metadata |
| `experiments/graph_swarm_v1/README_eng.md` | Architecture, API, diagnostic/reference semantics and interpretation limits |
| `experiments/graph_swarm_v1/README_rus.md` | Corresponding Russian documentation |
| `experiments/graph_swarm_v1/AMENDMENTS.md` | Authorized pre-training architecture specification update |
| `experiments/graph_swarm_v1/reports/stage2_report.md` | This report |

The containing commit hash is returned in the final response and can be retrieved
without embedding this report's own hash:

```bash
git log -1 --format=%H -- experiments/graph_swarm_v1/reports/stage2_report.md
```

## Final frozen architecture

Model specification version: `graph_swarm_v1:model:2`. Graph/cache protocol
remains `graph_swarm_v1:1`; split eligibility remains `stage1.1`.
The three main variants `mutation_self`, `graph_static`, `graph_swarm` use T=4;
the separately instantiated secondary `graph_swarm_t1` uses T=1.
No additional architectures were introduced.

| Component | Exact specification |
|---|---|
| Mutation encoding | Alphabet `ACDEFGHIKLMNPQRSTVWY`, 20 standard residues, WT≠MUT; concatenate WT one-hot, MUT one-hot, MUT−WT difference: q `[B,60]` |
| Frozen feature projection | Biased linear 384→64 followed by tanh |
| Local mutation injection | Biased 60→64→64 MLP, SiLU only after first linear; add ψ(q) only where immutable `seq_pos` matches the query's position |
| Graph message input | Receiver h_i, sender h_j, unchanged e_ij (17): 145 dimensions; Stage 1 source row 0, destination row 1 |
| Message MLP | 145→64→64; both biases true; SiLU only after first linear |
| Attention MLP | 145→32→1; first bias true, final scalar bias false; SiLU only after first linear |
| Graph aggregation | Stable incoming-neighbor softmax separately per destination/query; weighted sum; exactly zero for no incoming edges |
| Recurrent update | One biased `torch.nn.GRUCell(64,64)`, shared across residues/steps/queries; all messages from Hᵗ, then synchronous Hᵗ⁺¹ |
| Final readout | Concatenate mutation-row h, mean over all nodes, same q: 188; biased 188→64→1 MLP with SiLU only after first linear; scalar `[B]` |
| Excluded additions | Dropout=0; BatchNorm=false; LayerNorm=false; no extra residual or global vector inside recurrence; no external graph library |

Static/swarm have identical registered modules, parameter keys and tensor shapes;
strict state loading succeeds in both directions. Their only difference is
message refresh. Static computes mutation-conditioned m⁰ once after injection
and reuses the exact attached tensor at every step. It recomputes m⁰ on each
forward, with no detach or `no_grad`. Swarm recomputes both messages and attention
from current hidden states at each step. T=1 overrides of the primary variants
exist solely for equivalence diagnostics; the main comparison remains T=4.

Self messages are `(h_i,h_i,zeros(17))` through the same message MLP at each step.
`mutation_self` means **no cross-node exchange inside recurrent dynamics**.
Other residues still influence final predictions through the global mean. Its
attention module is registered and inactive, without changing the architecture
to equalize effectively used counts.

Forward takes one protein's tensor arrays and independent mutation-query arrays,
never labels or arbitrary target arguments. It validates shapes, finite floating
inputs, dtype/device agreement, the full immutable mapping, mutation ranges and
WT≠MUT, and endpoint/self-loop/duplicate validity. The existing Stage 1 record
and query validators retain geometric topology and WT-sequence alignment checks;
the tensor-only head does not infer WT sequence from contextual features.

`checkpoint_configuration()` provides the full frozen specification, variant,
actual steps and graph version separately from `state_dict`. A future Stage 3
loader must compare metadata as well as tensor keys/shapes: equal tensors alone
cannot distinguish static/swarm or T=1/T=4. No Stage 3 loader/pipeline was added.

Opt-in diagnostics return attached H⁰..Hᵀ, per-step messages/attention, q and
mutation rows. They preserve predictions, model tensors and parameter gradients.
The matched reference forward omits **only** local ψ(q) injection; frozen
features, graph, weights, q in the readout and model mode are matched. The
response definition is `delta_i^t = ||h_i,injected^t - h_i,reference^t||₂`.
This is diagnostic infrastructure, not WT molecular dynamics or mechanistic
evidence. Diagnostic tensors must not be mutated and retain autograd memory.

## Exact parameter counts

| Module | Registered trainable parameters |
|---|---:|
| Projection | 24,640 |
| Mutation MLP | 8,064 |
| Message MLP | 13,504 |
| Attention MLP | 4,704 |
| GRUCell | 24,960 |
| Readout MLP | 12,161 |
| **Total** | **88,033** |

| Variant | Steps | Total registered | Effectively used |
|---|---:|---:|---:|
| mutation_self | 4 | 88,033 | 83,329 |
| graph_static | 4 | 88,033 | 88,033 |
| graph_swarm | 4 | 88,033 | 88,033 |
| graph_swarm_t1 | 1 | 88,033 | 88,033 |

The two primary counts match exactly. Effective counts mean architectural use
in an ordinary injected forward, not a promise that every scalar gradient is
nonzero for every input. Attention can have zero effective influence when a
destination has just one incoming edge; this does not redefine model counts.

## Commands, counts and synthetic evidence

Commands ran from the repository root in the existing CPU `NeuroPP` environment:
Python 3.10.21, NumPy 2.2.6, PyTorch 2.5.1+cpu. No packages, weights or data were
installed/downloaded. Model tests temporarily use one CPU thread and restore
the previous setting afterwards.

**Before any edits**, Stage 1/1.1 baseline command:

```bash
PYTHONPATH=src conda run --no-capture-output -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -v
```

**Stage 2 in isolation**:

```bash
PYTHONPATH=src conda run --no-capture-output -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -p test_models.py -v
```

**Final combined Stage 1/1.1 + Stage 2 suite**:

```bash
PYTHONPATH=src conda run --no-capture-output -n NeuroPP python -m unittest discover -s tests/graph_swarm_v1 -v
```

| Run | Passed | Failed/errors | Skipped | unittest runtime |
|---|---:|---:|---:|---:|
| Before-edit baseline | 65 | 0 | 0 | 0.558 s |
| Stage 2 alone | 44 | 0 | 0 | 3.698 s |
| Final combined suite | **109** | **0** | **0** | **3.224 s** |

The 44 new tests include multiple variant/dtype/query subcases. Test counts are
unittest test methods, not the larger number of subcases.

| Synthetic evidence | Result |
|---|---|
| Encoding and initialization | Exact 60-vector components for all 20 amino acids; invalid indices/WT=MUT rejected; injection follows immutable positions under row permutations; independent query H⁰ |
| Incoming aggregation | Hand-computed directed fixture verifies receiver/sender order, per-destination/query groups, normalized sums, zero-degree messages; extreme logits remain finite; empty-edge singleton supported |
| Static/swarm T=1 | Predictions **and gradients of every active shared parameter tensor passed** in float32 and float64; separately instantiated swarm_t1 also matched |
| Static/swarm T=4 | Prescribed sender-dependent messages, injection and GRU weights produce different hidden states at distance 2 after step 2; first-step states agree |
| Refresh semantics | Static graph-message calls=1 per forward, swarm=4, swarm_t1=1; static message/attention tensor identity preserved across steps; new forwards recompute; m⁰ is mutation-conditioned |
| Permutation | Passed for all four variants in float32/float64 on tie-free Stage 1 topology, jointly permuting features/coordinates/seq_pos/residue UIDs/letters, remapping endpoints and shuffling matching edge attributes; predictions and trajectories agree |
| Batch independence | All variants pass batched-vs-separate predictions and states; changing query 0 does not affect query 1 |
| Swarm chain response | At t=0,1,2,3,4, response beyond distance t is zero within 1e-12; the frontier response is positive, detecting sequential updates or missing propagation |
| Static chain response | Response never expands beyond distance 1 for all four updates; mutated and first-neighbor nodes continue evolving relative to the matched reference |
| Self chain response | Non-mutated trajectories are exactly equal to reference throughout; final global-mean readout demonstrably uses other residues |
| Gradients and diagnostics | Finite, connected module-level gradients through projection/mutation/message/attention/GRU/readout; static message/attention gradients preserved; self attention gradients absent; diagnostics preserve predictions/state/parameter gradients |
| Serialization/configuration | In-memory state_dict roundtrip preserves predictions for all variants; complete explicit metadata and constructor constraints recorded |
| Input contract | Malformed shapes/mappings, nonfinite/dtype-invalid inputs, invalid endpoints/loops/duplicates rejected; labels/targets rejected |

General invariance/equivalence tolerances: float64 `atol=1e-8, rtol=1e-7`;
float32 `atol=1e-5, rtol=1e-5`. Chain response uses stricter float64 1e-12 bounds.
No test requires every individual scalar gradient to be nonzero. The T=4 fixture
is constructed deliberately; it does not assume random models must predict
differently.

**Tiny synthetic learning:** graph_swarm, four artificial mutation queries of
one four-node synthetic protein, prescribed targets `[-0.75,-0.25,0.25,0.75]`,
60 fixed Adam updates, learning rate 0.01, CPU float32. Initialization and frozen
features are deterministic. No hyperparameter search or validation set is used.

```text
initial MSE       = 0.31571337580680847
final MSE         = 0.000027170721295988187
final / initial   = 0.00008606135621132033
loss reduction    = approximately 99.9914%
```

The fixed engineering acceptance criteria (final/initial <0.05 and final MSE
<0.005) passed. This demonstrates trainability on this artificial fixture only;
it provides no MegaScale or real-protein accuracy evidence.

## Preservation and access boundaries

The following checks completed successfully (exit code 0):

```bash
git diff --check
git diff --exit-code --quiet 646b00819f435e6143a8e42af23de840c681d2e3 -- notebooks README.md README_rus.md README_eng.md results splits src/neuropp/graph.py src/neuropp/protocol.py src/neuropp/data_access.py src/neuropp/cache.py tests/graph_swarm_v1/test_infrastructure.py tests/graph_swarm_v1/test_provenance.py experiments/graph_swarm_v1/config.yaml experiments/graph_swarm_v1/reports/stage1_report.md experiments/graph_swarm_v1/reports/feature_provenance.json experiments/graph_swarm_v1/reports/graph_diagnostics.json experiments/graph_swarm_v1/reports/sequence_audit.json
```

A parsed-JSON comparison against the exact parent also confirmed that the only
changed existing protocol fields are `stage`, `stage_revision`, and `status`,
and the only added top-level field is `model`. All existing scientific/data
contract sections, including graph parameters, split, provenance, cache and
access policy, are identical. `stage1_excluded` remains the historical Stage 1
scope. Model architecture is a pre-training specification update, not a
performance-driven amendment.

- **No real mutation labels were accessed.** No MegaScale label file, isolated
  real development labels, or saved legacy mutation/error/prediction table was
  opened. Baseline access tests use synthetic temporary fixtures only; model
  learning targets are prescribed artificial numbers.
- **No protected evaluation occurred.** Neither head_holdout nor legacy_test
  labels were accessed. No validation MAE, real-data training, bootstrap,
  protected evaluator or unlock was implemented/run. Existing protection tests
  only exercise synthetic IDs and blocked/mocked readers.
- **Legacy files remain unchanged.** All notebooks, root READMEs and published
  results match the parent. Stage 1 source, tests, split manifest, config and
  audit reports remain unchanged. Existing feature caches, old head weights and
  upstream ProteinMPNN were neither loaded nor modified by this Stage 2 work.
- No real-artifact audits, real feature extraction or real graph preparation
  ran during Stage 2. No `03_mutation_graph_swarm.ipynb`, training pipeline,
  new edge/mutation descriptors or alternative architectures were added.

## Limitations and future evidence

Stage 1 already reported **all-node four-hop reachability** for its three
sampled real graphs, 1AOY/1E0L/1I6C (1.0000 fraction of other residues).
T=4 is therefore **not guaranteed-local**. Updates are neither physical
interaction shells nor physical time nor molecular signal propagation. The
scientific question remains whether refreshed evolving mutation-conditioned
messages help relative to fixed mutation-conditioned messages.

The chain response tests concern the **additional GraphSWARM head response at
fixed frozen features** and make no claim about ProteinMPNN's receptive field.
Synthetic sparse chains are testing fixtures, not changes to Stage 1's k=16
real graph topology. Global final readout includes all residues in every variant.

Only CPU float32/float64 were tested. GPU reduction-order behavior, large-protein
memory/performance, and real-data optimization are not established. Edge-level
intermediates scale with mutation batch size and edge count; opt-in attached
diagnostics additionally retain trajectories. Future engineering can assess
batch sizing without changing the frozen architecture.

**Future real-data evidence is absent.** No validation score, primary MAE
effect, model ranking, physical mechanism or generalization result was measured.
Stage 3 requires separate authorization for training/checkpoint handling and
must preserve protected-label access boundaries. Stage 2 is complete and stops
at this commit.

# NeuroPP: ThermoMPNN + SWARM

A pilot experiment to predict changes in protein stability caused by a single amino acid substitution. The experiment tested whether additional recurrent processing with a global summary of states improves predictions based on frozen ThermoMPNN features.

**Main result:** SWARM achieved the lowest mean test macro MAE, but a consistent advantage of repeatedly recomputing the global summary has not been established. The difference relative to the `static` control is small, and its 95% bootstrap interval includes zero.

## Setup and execution

Run the commands from the repository root. The environment and notebooks are configured for CPU execution.

```bash
conda env create -f environment.yml
conda activate NeuroPP

git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git external/ThermoMPNN
git -C external/ThermoMPNN checkout 370f76ec62bd929f7425e311d8df04a0d094990f

mkdir -p data/megascale
curl -fL https://zenodo.org/api/records/7992926/files/Processed_K50_dG_datasets.zip/content -o data/megascale/Processed_K50_dG_datasets.zip
curl -fL https://zenodo.org/api/records/7992926/files/AlphaFold_model_PDBs.zip/content -o data/megascale/AlphaFold_model_PDBs.zip
```

Run notebook cells sequentially using the interpreter from the `NeuroPP` environment. `01_baseline.ipynb` is an optional technical check; `02_swarm_experiment.ipynb` contains the full experiment. Required files are extracted from the archives automatically.

Trained head weights are not included in the repository. Re-evaluating the published models requires the original checkpoints; a new run performs 40 training runs. Before starting a new run, change `RUN_NAME` in the second notebook to preserve the published results in `results/swarm_final/`.

## 1. Task and sign convention

The input is the sequence and structure of the original protein (WT), the substitution position, and the new amino acid. The output is a ΔΔG prediction in kcal/mol.

Positive ΔΔG means **destabilization**; negative ΔΔG means stabilization:

$$
\Delta\Delta G = \Delta G_{\mathrm{unfold}}^{\mathrm{WT}} - \Delta G_{\mathrm{unfold}}^{\mathrm{MUT}}.
$$

In the MegaScale data, the target label is transformed as `ddg_exp = -ddG_ML`. A positive label corresponds to a mutant that requires less energy to unfold. The model learns to predict this effect from data.

## 2. ThermoMPNN

[ThermoMPNN](https://doi.org/10.1073/pnas.2314853121) uses knowledge from ProteinMPNN to predict stability. ProteinMPNN was originally trained to select an amino acid sequence compatible with a given three-dimensional structure (inverse folding).

[ProteinMPNN](https://doi.org/10.1126/science.add2187) represents the structure as a graph (nodes are residues; edges are connections between neighbors).

ThermoMPNN extracts these representations and passes them to a small output component that uses LightAttention and an MLP and was trained on stability data. In the configuration used here, each residue representation has 384 components: the last two decoder states with 128 components each and a 128-dimensional WT embedding.

The head produces 21 scores per position: for the 20 standard amino acids and the symbol `X`. For a substitution at position $i$, the prediction is computed as:

$$
\widehat{\Delta\Delta G}_{i,a\to b} = s_{i,b} - s_{i,a}.
$$

**Why it was chosen for the first pilot.** Open source code, pretrained weights, and a published data split. Structural features can be extracted once and frozen, and the small output head can be replaced. This allows a specific feature-processing mechanism to be studied with the available computational resources.

## 3. SWARM

[SWARM](https://arxiv.org/abs/1906.09400) is a recurrent network for processing sets of elements. In our case, one element, or agent, corresponds to an amino acid residue. It receives its own features $x_i$ and stores a hidden state $h_i$ and memory $c_i$. Trainable weights are shared across all residues and steps.

At each step, states are updated using the input features, previous states, and a global summary. In this pilot, the summary is the mean of the hidden states of the residues within one protein:

$$
\bar h_t = \frac{1}{L}\sum_{j=1}^{L}h_{j,t}.
$$

Input features remain constant throughout the forward pass. Initially, $h_i=c_i=0$; differences between states arise from differences between features. The first step processes residues individually; global information becomes available from the second step onward. During prediction, states are updated while weights remain fixed. This behavior matches the [authors' implementation](https://github.com/zalandoresearch/SWARM/blob/master/swarmlayer.py).

Why this mechanism is suitable for testing:

- It processes proteins of different lengths using a single shared cell.
- It allows a residue representation to be refined using a summary of the entire protein.
- Shared weights keep the model compact; at a fixed width, the cost of one update of the additional SWARM head grows linearly with the number of residues.
- It is equivariant to permutations of the precomputed feature rows (output rows are permuted correspondingly).

The mean summary compresses information and does not describe each residue pair separately.

The feature projection, three steps, and ΔΔG output are choices made for this pilot.

## 4. Experimental design

### Hypothesis and control comparisons

Hypothesis: repeatedly recomputing the global context helps estimate mutation effects more accurately than obtaining global information once, with a comparable architecture.

The pilot was exploratory: the set of controls and the number of seeds were refined on train/validation before test evaluation. There was no formal preregistration.

All new heads received a single matrix of frozen features $L\times384$ and were trained from scratch. They replaced the output component of ThermoMPNN. The entire original model remained frozen.

| Variant | Feature processing | Trainable parameters |
|---|---|---:|
| ThermoMPNN | Pretrained checkpoint; fixed reference | 0 |
| `mlp` | Independent processing of each residue: 384 → 32 → 256 → 21 | 26 165 |
| `self` | Recurrent head; the context branch receives the residue's own state | 26 101 |
| `static` | The global summary after the first update is used until the end of the forward pass | 26 101 |
| `swarm` | The global summary is recomputed between state updates | 26 101 |

In the recurrent variants, the projection, hidden state, and memory dimensions are 32; the number of updates is 3. The linear output head receives the concatenation of $h_i$ and $c_i$. In `static`, the gradient through the stored summary is not detached.

| Comparison | Question tested |
|---|---|
| ThermoMPNN → new heads | Is replacing the output component and training it anew beneficial? |
| `mlp` → `self` | Is recurrent processing beneficial compared with a simple head? |
| `self` → `static` | Is adding global information beneficial? |
| **`static` → `swarm`** | **Is repeatedly recomputing the global summary beneficial?** |

The last comparison is the primary one: the variants have the same parameter structure, number of steps, and initial weights for the same seed. With three steps, the difference arises at the third update: `static` uses the mean after the first step, while `swarm` uses the mean after the second. A comparison with the original ThermoMPNN alone does not isolate the effect of swarm information exchange.

### Data and checks

MegaScale provides experimental stability estimates from [Tsuboyama et al.](https://doi.org/10.1038/s41586-023-06328-6), obtained using cDNA display proteolysis. The data and AlphaFold structures are available on [Zenodo](https://zenodo.org/records/7992926).

The official ThermoMPNN split **by protein** was used: 239 train, 31 validation, 28 test. The authors' split was retained.

Single substitutions between standard amino acids with a finite numerical label were retained. The uniqueness of the WT sequence, its match with the PDB, the position range, the WT residue letter, and the full mutant sequence were checked. Single-chain structures with complete coordinates were used.

During preparation, predictions from the extracted features matched the standard ThermoMPNN call for five mutations of 1A32. In the first notebook, the MAE for these examples was 0.2383 kcal/mol.

### Training and final evaluation

- Four new heads were trained using seeds 42–51: **40 separate training runs**. For the recurrent variants, the same seed determined the same initial weights and order of train proteins.
- Adam optimizer, learning rate 0.001, 40 epochs. Each protein was processed once per epoch; one optimizer step used one protein and the mean MSE over its mutations. Proteins received the same number of steps, regardless of the number of mutations.
- For each head, the weights from the best epoch according to **validation macro MAE** were saved. Validation was used for checkpoint selection, so its results are not an independent final evaluation.
- The saved heads for all ten seeds were evaluated on test without retraining or selecting the best seed. Predictions were not combined into an ensemble.

The main metric is macro MAE:

$$
\mathrm{MAE}_p = \frac{1}{n_p}\sum_{k=1}^{n_p}|\widehat y_{p,k}-y_{p,k}|,
\qquad
\mathrm{macro\ MAE} = \frac{1}{P}\sum_{p=1}^{P}\mathrm{MAE}_p.
$$

It gives each protein equal weight. Pooled MAE and RMSE over all mutations were also calculated. Means and standard deviations across seeds describe the errors of individual trained models.

To assess uncertainty in the differences, MAE was first averaged across seeds within each test protein. A paired bootstrap **over whole proteins** was then performed: 10 000 samples drawn with replacement from 28 proteins, RNG 2026, with the same samples used for all comparisons. The bounds of the 95% interval are the 2.5% and 97.5% percentiles. Mutations within one protein were not treated as independent observations.

### Expected result

Support for the primary hypothesis would have been a positive `MAE(static) − MAE(swarm)`, consistent across seeds and proteins, with an interval excluding zero. An improvement over ThermoMPNN without an advantage over `static` would not have allowed the gain to be attributed to repeated information exchange.

## 5. Results

### Validation

All error values are in kcal/mol. Std is the standard deviation across ten seeds; the original ThermoMPNN was evaluated once.

| Model | Mean macro MAE | Std |
|---|---:|---:|
| ThermoMPNN | 0.430518 | — |
| `mlp` | 0.411275 | 0.002213 |
| `self` | 0.404478 | 0.002089 |
| `static` | **0.403343** | 0.001880 |
| `swarm` | 0.403562 | 0.001880 |

The best mean result belongs to `static`. The mean difference `static − swarm` is **−0.000219**: SWARM performs better for four seeds and worse for five; for one seed, the difference is numerically close to zero. The initial positive mean across three seeds did not persist after expansion to ten.

### Test

| Model | Mean macro MAE | Std | Mean pooled MAE | Mean pooled RMSE |
|---|---:|---:|---:|---:|
| ThermoMPNN | 0.529062 | — | 0.534775 | 0.717473 |
| `mlp` | 0.524161 | 0.005092 | 0.528500 | 0.713910 |
| `self` | 0.515525 | 0.002670 | 0.520109 | 0.710171 |
| `static` | 0.514592 | 0.004307 | 0.519826 | 0.707061 |
| `swarm` | **0.513064** | 0.005217 | **0.518153** | **0.706672** |

SWARM has the lowest mean test error. The reduction in macro MAE relative to ThermoMPNN is **0.015998 kcal/mol, approximately 3.02%**; relative to `static`, it is **0.001528 kcal/mol, approximately 0.30%**. The percentages refer to error reduction, not to a change in protein stability.

In paired comparisons across seeds, SWARM performs better than `static` in seven runs and worse in three. Seed 46 produced a difference of 0.014028 kcal/mol—approximately 92% of the summed difference across ten runs. The mean gain for the remaining nine is approximately 0.00014 kcal/mol. This is a sensitivity analysis; the primary result includes all ten seeds.

After averaging MAE across seeds, SWARM performs better than `static` on **15 of 28 proteins** and worse on 13; the median gain is 0.001033 kcal/mol. The effect is heterogeneous: the largest gain on 2BTH is +0.018529, and the largest deterioration on HEEH_KT_rd6_0793 is −0.015078 kcal/mol.

### Uncertainty in the differences

A positive difference indicates an advantage for the second variant.

| MAE comparison | Mean difference | Lower bound of 95% CI | Upper bound of 95% CI |
|---|---:|---:|---:|
| ThermoMPNN − `swarm` | 0.015998 | −0.011507 | 0.043193 |
| `self` − `static` | 0.000933 | −0.005867 | 0.007352 |
| **`static` − `swarm`** | **0.001528** | **−0.001770** | **0.004901** |
| `self` − `swarm` | 0.002461 | −0.004006 | 0.009279 |

**All intervals include zero.** This analysis does not establish the direction of any of the listed differences with confidence.

## 6. Discussion and limits of the conclusions

The new heads produced a lower mean test error than the original ThermoMPNN. However, this difference combines the effect of the head architecture and its new training.

In the primary comparison between `static` and `swarm`, the effect is small, the model ranking differs between validation and test, and the test gain is sensitive to one seed. In this configuration, repeatedly recomputing the context did not show a convincing, consistent advantage (the features may already have contained enough context, the mean summary may have been limiting, or the effect may have been too small for the available sample).

Limitations of the pilot:

- One configuration was tested: frozen features, 32-dimensional states, and three updates.
- Test contains 28 proteins.
- The bootstrap describes variation across proteins conditional on a fixed panel of trained models (it does not jointly include uncertainty from new training and experimental labels; possible dependence among related proteins was not modeled separately).
- One stability dataset was used; transfer to other protein types, experiments, or conditions was not tested. Functional effects of mutations and protein dynamics were not evaluated.

## 7. Conclusions

1. The combination of frozen ThermoMPNN features with new MLP and recurrent heads is functional; consistency of the inputs and original predictions was verified.
2. SWARM achieved the best mean numerical result on test, but its advantage over `static` and the original ThermoMPNN is not supported by the calculated intervals.
3. The primary hypothesis that repeatedly recomputing the global summary is beneficial **did not receive convincing support in this pilot**. The result is limited to the chosen configuration and data.

## References

1. Dieckhaus H., Brocidiacono M., Randolph N. Z., Kuhlman B. **Transfer learning to leverage larger datasets for improved prediction of protein stability changes.** PNAS, 2024. [Paper](https://doi.org/10.1073/pnas.2314853121), [ThermoMPNN code](https://github.com/Kuhlman-Lab/ThermoMPNN).
2. Dauparas J. et al. **Robust deep learning–based protein sequence design using ProteinMPNN.** Science, 2022. [Paper](https://doi.org/10.1126/science.add2187).
3. Vollgraf R. **Learning Set-equivariant Functions with SWARM Mappings.** arXiv:1906.09400, 2019. [Paper](https://arxiv.org/abs/1906.09400), [SWARM implementation](https://github.com/zalandoresearch/SWARM).
4. Tsuboyama K. et al. **Mega-scale experimental analysis of protein folding stability in biology and design.** Nature, 2023. [Paper](https://doi.org/10.1038/s41586-023-06328-6), [data](https://zenodo.org/records/7992926).

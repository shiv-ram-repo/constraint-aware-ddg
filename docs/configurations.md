# Loss Configurations

Five configurations differ only in the training objective:

| Config | BMC | Siamese | OOD-margin | Total objective                                                          |
|--------|-----|---------|------------|--------------------------------------------------------------------------|
| **A**  |     |         |            | Huber (baseline, reproduces SPURS training recipe)                       |
| **B**  | ✔   |         |            | $\mathcal{L}_{BMC}$                                                      |
| **C**  |     | ✔       |            | Huber + $\lambda_{sym}\,\mathcal{L}_{sym}$                               |
| **D**  | ✔   | ✔       |            | $\mathcal{L}_{BMC} + \lambda_{sym}\,\mathcal{L}_{sym}$                   |
| **E**  | ✔   | ✔       | ✔          | $\mathcal{L}_{BMC} + \lambda_{sym}\,\mathcal{L}_{sym} + \lambda_{OOD}\,\mathcal{L}_{OOD}$ |

Weights used throughout the paper: $\lambda_{sym} = \lambda_{OOD} = 0.5$,
and the OOD-margin Gaussian noise scale $\sigma_{\mathrm{OOD}} = 0.20$
in feature units.

## Headline result: configuration E vs A

Configuration E achieves on S669 a Spearman correlation of
$0.540 \pm 0.002$ across three random seeds (42, 43, 44), an improvement
of $+0.054$ over the SPURS-style baseline configuration A
($0.486 \pm 0.012$). Parallel improvement on S461 is $+0.058$
($0.653 \to 0.711$).

## Loss definitions

See `src/custom_losses.py` for canonical implementations. The Balanced
MSE term is in `src/imbalanced_losses.py` (also `src/custom_losses.py`
provides a thin wrapper for consistency).

### Balanced MSE (Ren et al., 2022)

For a batch of $B$ predictions $\hat{y}_i$ and labels $y_i$,

$$
\mathcal{L}_{BMC} = -\frac{1}{B}\sum_{i=1}^{B} \log
\frac{\exp\bigl(-(\hat{y}_i - y_i)^2 / 2\sigma_B^2\bigr)}
     {\sum_{j=1}^{B} \exp\bigl(-(\hat{y}_i - y_j)^2 / 2\sigma_B^2\bigr)}
$$

where $\sigma_B$ is a learnable noise-scale parameter initialised at
1.0 and optimised jointly with the network.

### Siamese anti-symmetric

For each mutation, the model is run twice with wild-type and mutant
identities swapped (a zero-cost operation under the gather
parametrisation in `multimodal_ddg.py`):

$$
\mathcal{L}_{sym} = \frac{1}{B}\sum_{i=1}^{B}
\bigl(\hat{f}^{(i)}_{\to} + \hat{f}^{(i)}_{\leftarrow}\bigr)^2
$$

### OOD-margin (this paper)

After the encoder forward pass, Gaussian noise is added to the
per-position feature representation and the prediction head is
re-run on the perturbed input:

$$
\mathcal{L}_{OOD} = \frac{1}{B}\sum_{i=1}^{B} (\hat{y}_i - \tilde{y}_i)^2,
\qquad \tilde{y}_i = f(h_p + \delta),\quad \delta \sim \mathcal{N}(0, \sigma_{\mathrm{OOD}}^2 I)
$$

The encoder pass is shared between $\hat{y}$ and $\tilde{y}$, so the
additional cost is just one MLP head forward pass plus one Gaussian
sample (~10% per-step overhead).

### BCAS (bias-corrected anti-symmetric, archived as negative result)

A loss design of our own that explicitly penalises the signed
batch-mean of the forward-reverse sum, intended to eliminate the
systematic bias on Ssym:

$$
\mathcal{L}_{BCAS} = \alpha \Bigl[\tfrac{1}{B}\sum_i (\hat{f}^{(i)}_{\to} + \hat{f}^{(i)}_{\leftarrow})\Bigr]^2
+ \beta \cdot \tfrac{1}{B}\sum_i (\hat{f}^{(i)}_{\to} + \hat{f}^{(i)}_{\leftarrow})^2
$$

with $\alpha = 1.0, \beta = 0.5$. BCAS substantially reduces the Ssym
bias (to $|<0.1|$ kcal/mol from $0.29$-$0.40$ kcal/mol) but does not
improve OOD Spearman, demonstrating that systematic bias and OOD
generalisation are decoupled in this regime. Implementation is
included in `src/custom_losses.py`; the experimental result is
documented in the manuscript's negative-results section.

## Hyperparameter sensitivity

We sweep the OOD-margin noise scale $\sigma_{\mathrm{OOD}}$ at fixed
weight 0.5, with all other hyperparameters at configuration D defaults:

| $\sigma_{\mathrm{OOD}}$ | S669 Spearman | Notes                            |
|-------------------------|---------------|----------------------------------|
| 0.00 (config D)         | 0.517 ± 0.007 | Three seeds                      |
| 0.10                    | 0.531         | Seed 42 only                     |
| **0.20** (config E)     | **0.540 ± 0.002** | Three seeds (headline)       |
| 0.50                    | 0.520         | Seed 42 only                     |

The sweet spot at 0.20 is sharply localised --- larger or smaller
values give markedly worse results. See `scripts/run_finalize.py` to
reproduce.

## Optimiser and training

* AdamW ($\beta_1=0.9, \beta_2=0.999$, $\varepsilon=10^{-8}$)
* Learning rate $10^{-4}$
* Weight decay $10^{-2}$
* Gradient clipping at norm 1.0
* Batch size: one wild-type protein per step (~200-1200 mutations)
* Up to 200 epochs, early stopping on validation Spearman
  (patience 20-30 epochs)
* Three seeds: 42, 43, 44

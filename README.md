# ADHD-dual-stream-pipeline-execution-plan
Now I have the full picture. Let me give you a comprehensive, expert-level execution plan.Here's your complete domain-expert execution plan. Let me break this down clearly:---

## The full expert-level plan

### What you're actually dealing with

You have two documents: (1) a research proposal describing what to build, and (2) a reviewer's list of 18+ specific errors in an already-written draft. This means **someone already generated a paper draft with mistakes**. Your job is to build the correct pipeline AND fix those specific issues.

---

### Critical issues to fix (from `Suggested_Revision.docx`)

These are the errors the reviewer flagged — you must address all of them in your implementation:

**Methods errors:**
- The sign matrix allocation rationale is never explained → add a clear sentence justifying why magnitude goes to resistance stream and sign goes to GNN
- Pseudoinverse tolerance not specified → always use `np.linalg.pinv(L, rcond=1e-6)`
- MDS used for "visualization" but claimed as features → MDS produces node coordinates; you then take the top-K eigenvalues as subject-level features by averaging or flattening — explain this explicitly
- CNN on a symmetric matrix is unjustified → cite at least one paper showing spectral patterns in FC matrices benefit from spatial convolution
- Feature concatenation without dimensionality reduction → use PCA or just restrict to top-K before concatenating
- MST + additional edges: never specified how many extra edges → fix to k=50 and justify with a sensitivity analysis
- Forman-Ricci classified as "topological" → it is geometric; add one sentence calling it "geometry-enhanced"
- Node features for GNN never defined → specify exactly what they are (sign FC row of each node = 200-dim vector)
- Cross-modal attention formula missing → provide the explicit QKV formula in the paper
- Hyperparameters not listed → document batch size (32), LR (1e-4), epochs (100), early stopping (patience=10)

**Results errors:**
- "Median ± SD" is statistically incorrect → use median (IQR) or mean ± SD, not median ± SD
- Eigenvalues localized to brain regions → this is wrong; eigenvalues are global; remove that claim
- Table 12 feature importance sums to 135.7% → fix to use normalized SHAP values that sum to 100%
- P-values without multiple comparisons correction → use FDR correction (Benjamini-Hochberg)
- Subtype clinical score differences reported without ANOVA → run one-way ANOVA with post-hoc Tukey

---

### Day-by-day execution plan

**Day 1 — Data and environment setup**

Install: `torch`, `torch-geometric`, `gudhi` or `giotto-tda`, `GraphRicciCurvature`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`

The Kaggle dataset has a `TRAIN_NEW.tsv` and `TEST_NEW.tsv` where each row is a subject and columns are the 19,900 upper-triangle values of the 200×200 FC matrix. Load these into numpy arrays, reconstruct the full symmetric matrix per subject, apply Fisher Z-transform, then decompose into `|FC|` (for resistance) and `sign(FC)` (for GNN edge signs). Handle NaNs by replacing with column median.

**Day 2 — Resistance stream**

```python
import numpy as np
from scipy.linalg import pinv

def resistance_matrix(FC_abs):
    deg = FC_abs.sum(axis=1)
    L = np.diag(deg) - FC_abs
    Lp = pinv(L, rcond=1e-6)  # Moore-Penrose pseudoinverse
    n = L.shape[0]
    Omega = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Omega[i,j] = Lp[i,i] + Lp[j,j] - 2*Lp[i,j]
    return Omega
```

Then double-center Omega to get a Gram matrix → `eigh` → take top-K=50 eigenvalues+eigenvectors as spectral features. Separately feed the 200×200 Omega as a single-channel image into a 3-layer CNN.

**Day 3 — Topology stream**

For persistent homology use `giotto-tda`'s `VietorisRipsPersistence` on the distance matrix `d = 1 - |FC|`. Extract H0 and H1 landscapes with 100 sample points each. For the GNN, build a sparse graph per subject using MST + top-50 highest-weight edges. Node features = 200-dim sign FC row for each node. Compute Forman-Ricci curvature using the `GraphRicciCurvature` library (specifically `FormanRicci`) and assign curvature as edge weights.

**Day 4 — Dual-stream fusion**

The cross-modal attention formula to include in the paper:

> `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`
> where Q = resistance embedding (d=128), K = V = topology embedding (d=128)

Train with 5-fold CV, Adam optimizer, lr=1e-4, batch=32, early stopping patience=10, max 100 epochs. Report mean ± std of AUC, F1, and accuracy across folds for all models.

**Day 5 — Subtype analysis**

Extract the penultimate-layer 256-dim embedding for each training subject. Run GMM with k=2,3,4,5 and use BIC to select optimal k. For each subtype: run one-way ANOVA across CBCL and SWAN clinical scores with Tukey post-hoc correction. Visualize using t-SNE and the top differentially connected FC edges (by Cohen's d).

---

### Paper fixing checklist

Before submitting, verify:
- Table 12 SHAP values sum to exactly 100%
- All p-values state "FDR-corrected (Benjamini-Hochberg)"
- Dataset characteristics use "mean ± SD" or "median (IQR)" — not "median ± SD"
- Curvature is called "geometry-enhanced" not "topological" in the text
- The dual-stream framing is consistent everywhere (curvature is inside the topology stream, not a third stream)
- All hyperparameters are listed in a table in the Methods section
- The cross-modal attention QKV formula appears in the Methods section

---

### What to deliver by March 30

Minimum viable deliverable for your Monday presentation: `data_prep.ipynb` + `resistance_stream.ipynb` + `topology_stream.ipynb` with real numbers from at least one fold of cross-validation. Even partial AUC results from Stage 2 alone (resistance features + baseline SVM) give you something concrete to show. The full dual-stream fusion can follow in the next few days.


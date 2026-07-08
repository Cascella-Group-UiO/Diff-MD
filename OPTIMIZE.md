# Gradient Methods in ∂-HyMD

This document describes the gradient computation methods available in ∂-aMD
for differentiable molecular dynamics parameter optimization.  It covers the
mathematical foundations, implementation details, and practical guidance for
choosing between methods.

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
  - [Reverse-Mode AD (VJP)](#reverse-mode-ad-vjp)
  - [Forward-Mode AD (JVP)](#forward-mode-ad-jvp)
  - [Central Finite Differences](#central-finite-differences)
- [MPI Allreduce Semantics](#mpi-allreduce-semantics)
- [Memory and Computational Complexity](#memory-and-computational-complexity)
- [When to Use Each Method](#when-to-use-each-method)
- [Configuration](#configuration)
- [Validation](#validation)
- [Loss Functions](#loss-functions)
- [Metrics](#metrics)
  - [mse](#mse)
  - [rmse](#rmse)
  - [smape](#smape)
  - [l2e](#l2e)
  - [Kullback_Leibler](#kullback_leibler)
  - [wasserstein_1d](#wasserstein_1d)
- [References](#references)


## Overview

∂-HyMD optimizes Lennard-Jones parameters (σ, ε) by differentiating a loss
function through a full molecular dynamics trajectory.  The gradient of the
loss with respect to the trainable parameters can be computed in three ways:

| Method | Key | Gradient | Memory | Exactness |
|--------|-----|----------|--------|-----------|
| Reverse-mode AD | `reverse` | `value_and_grad` | O(n_steps) | Exact |
| Forward-mode AD | `jvp` | `jax.jvp` | **O(1)** | Exact |
| Finite differences | `finite_diff` | Central FD | **O(1)** | Approximate |

All three methods produce the same global gradient (validated to <10⁻⁶
relative error for JVP vs. reverse-mode).  The key difference is **memory
scaling**: reverse-mode stores the forward-pass carry across all time steps
during backpropagation, while JVP and finite differences only need the
current time step.


## Mathematical Background

### Reverse-Mode AD (VJP)

Given a scalar loss function L(θ) : ℝⁿ → ℝ composed through a simulation
of T time steps, reverse-mode automatic differentiation computes the full
gradient ∇_θ L in a single backward pass by applying the chain rule from
output to input:

```
∇_θ L = (∂L/∂x_T) · (∂x_T/∂x_{T-1}) · ... · (∂x_1/∂θ)
```

This is the **vector-Jacobian product** (VJP): a row vector (the cotangent)
is propagated backward through each operation.  For a scalar loss, one
backward pass yields the entire gradient regardless of the number of
parameters n.

**Cost:** 1 forward pass + 1 backward pass ≈ (1 + α) × forward, where
α ≈ 2–8 depending on the complexity of the backward pass.

**Memory:** The backward pass requires access to intermediate values from
the forward pass.  With `jax.checkpoint` (rematerialization), per-step
internal activations are recomputed rather than stored, reducing memory to
O(1) for internal state.  However, the **carry** of `lax.scan` (positions,
velocities, forces, energies — 19 arrays totaling ~390 KB per step) must be
stored for all T steps, giving **O(T) total memory** for the carry stack.

At T = 2000 steps this requires ~780 MB; at T = 4000 steps the ~1.56 GB
contiguous allocation fails in the fragmented BFC memory pool, causing
out-of-memory (OOM) errors even on 96 GB GPUs.


### Forward-Mode AD (JVP)

Forward-mode AD computes the **Jacobian-vector product** (JVP): a tangent
vector is propagated forward alongside the primal computation.  For a
single tangent direction eᵢ (the i-th standard basis vector):

```
(L(θ), ∂L/∂θᵢ) = jvp(L, θ, eᵢ)
```

This yields one component of the gradient per forward pass.  For n trainable
parameters, n forward passes are needed to reconstruct the full gradient:

```
∇_θ L = [jvp(L, θ, e₁), jvp(L, θ, e₂), ..., jvp(L, θ, eₙ)]
```

**Why JVP and VJP produce identical gradients:**

Both methods compute the same mathematical object — the Jacobian matrix
J = ∂L/∂θ — decomposed differently:

```
VJP:  vᵀ · J        (row extraction — one pass for all n columns)
JVP:  J · eᵢ        (column extraction — one pass per column)
```

For a scalar loss (1×n Jacobian = the gradient), VJP extracts the entire
row in one pass, while JVP extracts one column per pass.  Both use the
same chain rule through the same computation graph with the same floating-
point precision.  **There is no approximation — the gradient is exact to
machine precision** (validated: max relative error < 10⁻⁶ in float32).

**Cost:** n forward passes (primal + tangent), each approximately 2× the
cost of a bare forward pass due to dual-number arithmetic.

**Memory:** The tangent is accumulated alongside the primal in the same
`lax.scan` loop.  No backward pass means **no carry stack** — memory is
**O(1) regardless of T**.  This is the critical advantage: JVP can run
simulations of arbitrary length without OOM.


### Central Finite Differences

The gradient is approximated numerically:

```
∂L/∂θᵢ ≈ [L(θ + εᵢeᵢ) - L(θ - εᵢeᵢ)] / (2εᵢ)
```

where εᵢ = ε_rel · max(|θᵢ|, 10⁻⁶) is a relative step size.

**Cost:** 2n + 1 forward passes (1 primal + 2 per parameter).

**Memory:** O(1) — each evaluation is an independent forward pass.

**Accuracy:** The truncation error is O(ε²), giving ~4 correct digits
with ε_rel = 10⁻⁴.  This is sufficient for optimization (the learning
rate noise dominates), but less precise than AD methods.  The gradient
can also be affected by numerical noise in the simulation for small ε.


## MPI Allreduce Semantics

The loss functions in ∂-aMD contain `mpi4jax.allreduce(obs, op=MPI.SUM)`
to aggregate observables across MPI ranks.  The three gradient methods
interact with this collective differently:

### Reverse-mode (VJP)
The VJP rule for `allreduce(SUM)` is the **identity** — the cotangent
passes through unchanged without MPI communication.  This means
`value_and_grad` returns a **rank-local** gradient that must be explicitly
allreduced afterward to obtain the global gradient.

### Forward-mode (JVP)
The JVP rule for `allreduce(SUM)` is `allreduce(SUM)` of the tangent
(confirmed in mpi4jax source: `ad.primitive_jvps[mpi_allreduce_p]`).
This means `jax.jvp` returns an **already-global** gradient.  No
post-hoc allreduce is needed — applying one would double-count.

### Finite differences
Each L(θ ± ε) evaluation runs the full loss including allreduce, so the
finite-difference gradient is inherently **global**.  No post-hoc
allreduce is needed.

**Implementation consequence:** The allreduce block in the `step()`
function is only applied for `grad_method = "reverse"`.  For `"jvp"` and
`"finite_diff"`, the gradient is used directly.


## Memory and Computational Complexity

For n trainable parameters and T simulation steps:

| Property | Reverse | JVP | Finite Diff |
|----------|---------|-----|-------------|
| Forward passes | 1 | n + 1 | 2n + 1 |
| Backward passes | 1 | 0 | 0 |
| Total cost (×fwd) | 1 + α | ~2n + 1 | 2n + 1 |
| Memory (carry) | O(T) | **O(1)** | **O(1)** |
| Gradient quality | Exact | Exact | O(ε²) approx |
| MPI allreduce after? | Yes | No | No |

With α ≈ 8 (measured) and n = 4 parameters:

| Method | Effective cost | Can run T=4000? |
|--------|---------------|-----------------|
| Reverse | 9 × fwd | **No** (OOM) |
| JVP | 9 × fwd | **Yes** |
| Finite Diff | 9 × fwd | **Yes** |

The crossover where reverse-mode becomes more efficient than JVP occurs at:

```
n_params > α ≈ 8
```

Below this threshold (which covers the current ∂-aMD use cases of 4–8 LJ
parameters), JVP is competitive in speed and strictly superior in memory.


## When to Use Each Method

| Scenario | Recommended | Why |
|----------|-------------|-----|
| n_params ≤ 8, any n_steps | `jvp` | O(1) memory, exact gradients, no OOM |
| n_params ≤ 8, n_steps ≤ 1000 | `reverse` | Slightly faster (one backward vs n forward) |
| n_params > 8 | `reverse` | O(1) in n_params vs O(n) for JVP |
| Debugging / validation | `finite_diff` | Independent numerical check |
| Production, long trajectories | `jvp` | Enables picosecond-scale training |


## Configuration

### TOML Configuration

Add `grad_method` to the `[nn]` section of your training TOML file:

```toml
[nn]
systems = ["."]
n_epochs = 50
grad_method = "jvp"        # Options: "reverse" (default), "jvp", "finite_diff"
fd_epsilon = 1e-4           # Only used when grad_method = "finite_diff"

# ... rest of [nn] section unchanged ...
```

### Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `grad_method` | string | `"reverse"` | Gradient computation method |
| `fd_epsilon` | float | `1e-4` | Relative step size for finite differences |
| `clear_xla_cache` | int | `0` | Interval (in epochs) for calling `jax.clear_caches()`. `0` = disabled. Set to e.g. `5` on systems with low `vm.max_map_count` (like Betzy: 65 530) to prevent VMA exhaustion |

### Valid values for `grad_method`

- **`"reverse"`** — Standard reverse-mode AD via `jax.value_and_grad`.
  The original method.  Best when n_params > 8 or n_steps is small.

- **`"jvp"`** — Forward-mode AD via `jax.jvp`.  One forward pass per
  parameter.  O(1) memory.  **Recommended for long simulations** with
  up to ~8 trainable parameters.

- **`"finite_diff"`** — Central finite differences.  Approximate gradients
  via (L(θ+ε) − L(θ−ε)) / 2ε.  Useful as an independent validation
  method.  Accuracy controlled by `fd_epsilon`.

### Example: enabling JVP for long-trajectory training

```toml
[nn]
systems = ["."]
n_epochs = 100
grad_method = "jvp"
teacher_forcing = false
equilibration = 0

[nn.optimizer]
name = "adam"
learning_rate = 0.01

[nn.loss]
name = "coordination_distance"
metric = "mse"
dist_weight = 1.0
constraint = "harmonic"
k_constraint = 0.01
```

### Backward compatibility

If `grad_method` is not specified, the default `"reverse"` is used,
preserving the original behavior with no changes to existing TOML files.


## Validation

The three gradient methods were cross-validated on a 2LUA zinc
metalloprotein system (15,725 atoms, 4 trainable LJ parameters,
4 MPI ranks, A100 GPUs):

### At n_steps = 200 (Pass 1, cached runtime)

| Parameter | Reverse (global) | JVP | |JVP−Rev|/|Rev| |
|-----------|-----------------|-----|----------------|
| σ_SZ | 0.09079529 | 0.09079529 | 8.2 × 10⁻⁸ |
| ε_SZ | 0.05107779 | 0.05107785 | 1.2 × 10⁻⁶ |
| σ_Zn  | 0.00208783 | 0.00208783 | 3.4 × 10⁻⁷ |
| ε_Zn  | 0.08801268 | 0.08801265 | 3.4 × 10⁻⁷ |

**JVP matches reverse-mode to < 10⁻⁶ relative error** (float32 precision
limit), confirming the two methods produce mathematically identical
gradients.

### Timing (n_steps = 200, 4 MPI ranks, A100)

| Method | Time | Ratio |
|--------|------|-------|
| Reverse (value_and_grad) | 324.6s | 1.00× |
| JVP (4 params) | 719.2s | 2.22× |
| Finite diff (9 evals) | 94.5s | 0.29× |

JVP is ~2× slower per epoch at n_steps = 200, but this comparison is
irrelevant in practice: **reverse-mode cannot run at n_steps > 2000**
(OOM), while JVP runs at any n_steps with constant memory.


##ADD!
Peeks at the training TOML for double_precision = true in [nn] section to run the optimize 
calculation in double precision  (positions and velocity)


## Loss Functions

This section documents the available loss functions, their purpose, and the
optional parameters that control their behavior.  All distribution-based loss
functions now use a **fixed-width Gaussian KDE** with absolute bandwidth

$$
h = \text{width\_ratio} \times \text{bin\_size}
$$

where `bin_size` is the spacing of the evaluation grid.  This means
`width_ratio` has the same meaning across systems and does **not** depend on
the instantaneous sample variance of the simulated trajectory.

### `density_and_apl`

**Purpose:** Lipid membrane optimization.  Matches the lateral number density
profile and the area per lipid (APL) of a simulated membrane against
reference data.

**How it works:**
1. Runs a differentiable MD simulation
2. For each frame, computes the center of mass along z (using circular mean
   to handle PBC), centers particles in the box, then builds a per-type
   lateral density profile via KDE with PBC reflection
3. Averages the KDE density across frames and MPI ranks
4. Computes the error as a weighted sum of density MSE (per type, averaged
   over types) and APL error

**System-specific arguments** (set in `[nn.system_args.<name>]`):

| Key | Type | Description |
|-----|------|-------------|
| `target_density` | string | Path to `.npy` or `.xvg` file with reference density (first row = z_range, rest = density per type) |
| `com_type` | string | Bead type name used for centering (e.g. `"C4A"` for tail beads) |
| `n_lipids` | int | Total number of lipids in the system (both leaflets) |
| `target_apl` | float | Target area per lipid (nm²) |

**General loss arguments** (set in `[nn.loss]`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `metric` | string | — | Error metric: `"mse"`, `"rmse"`, `"smape"`, `"l2e"` |
| `density_weight` | float or list | `1.0` | Weight for density error (scalar or per-type array) |
| `apl_weight` | float | `1.0` | Weight for APL error |
| `width_ratio` | float | `1.0` | Absolute KDE width in grid units: `width_ratio * bin_size`. Increase for smoother profiles |
| `k_constraint` | float | `0.01` | Strength of harmonic/cubic parameter constraint |
| `constraint` | string | `None` | `"harmonic"` or `"cubic"` — restrains trainable parameters toward initial values |
| `upper_boundary` | float | `None` | Soft upper bound on epsilon values (sigmoid penalty). Old key: `boundary` |
| `lower_boundary` | float | `None` | Soft lower bound on epsilon values |
| `boundary_C` | float | `500` | Amplitude of boundary penalty sigmoid |
| `boundary_S` | float | `2` | Steepness of boundary penalty sigmoid |


### `radius_of_gyration`

**Purpose:** Polymer/protein CG optimization.  Matches the mean radius of
gyration (Rg) of chain molecules against a target value.

**How it works:**
1. Runs a differentiable MD simulation
2. For each frame (via `vmap`), computes the mass-weighted Rg for each chain
   and averages over chains
3. Sums Rg over frames, allreduces over MPI ranks, and divides by
   `(ranks × frames)` to get the global mean Rg
4. Computes the error as `rg_weight * metric(mean_Rg, target_rg)`

**System-specific arguments:**

| Key | Type | Description |
|-----|------|-------------|
| `target_rg` | float | Target radius of gyration (nm) |
| `n_chains` | int | Number of polymer chains in the system |
| `resname` | string | Residue name used to identify chain atoms |

**General loss arguments:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rg_weight` | float | `1.0` | Weight for Rg error |
| Other boundary/constraint keys | — | — | Same as `density_and_apl` |


### `radius_of_gyration_dist`

**Purpose:** Like `radius_of_gyration` but matches the full probability
distribution of Rg values (instead of just the mean).

**How it works:**
1. Collects a time series of Rg values (one per frame, via `vmap`)
2. Builds a fixed-width Gaussian KDE of the Rg distribution using `width_ratio * bin_size`
3. Compares the KDE against a reference distribution

**Additional system-specific arguments:**

| Key | Type | Description |
|-----|------|-------------|
| `target_dist` | string | Path to `.npy` or `.xvg` file with reference Rg distribution |

**Additional general arguments:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `width_ratio` | float | `1.0` | Absolute KDE width in grid units: `width_ratio * bin_size` |


### `radius_of_gyration_median`

**Purpose:** Robust variant that uses the **median** Rg across MPI ranks
instead of the mean.  Reduces sensitivity to outlier replicas.


### `radius_of_gyration_filter_repls`

**Purpose:** Filters out MPI replicas whose mean Rg deviates by more than
10% from the median, then computes the loss on the remaining replicas.
Useful for multi-GPU runs where some replicas may explore unphysical
conformations.


### `radius_of_gyration_and_end_to_end`

**Purpose:** Matches both the Rg and the end-to-end distance of chain
molecules.

**Additional arguments:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `target_end_to_end` | float | — | Target end-to-end distance (nm) |
| `rg_weight` | float | `10.0` | Weight for Rg error |
| `end_to_end_weight` | float | `1.0` | Weight for end-to-end error |


### `coordination_distance_dist`

**Purpose:** Atomistic protein optimization.  Matches pairwise distance
distributions between coordinating atom groups (e.g. ligand–metal distances).

**How it works:**
1. For each coordination pair, computes all pairwise distances (minimum-image
   PBC) across all frames using `vmap`
2. Builds a fixed-width Gaussian KDE per pair using `width_ratio * bin_size`
3. Compares against reference distributions

**System-specific arguments:**

| Key | Type | Description |
|-----|------|-------------|
| `coord_pairs` | list[list[str, str]] | Pairs of atom names, e.g. `[["SZ", "Zn"], ["NZ", "Zn"]]` |
| `target_coord_dist` | string | Path to `.xvg` file with reference distributions |


### `coordination_distance`

**Purpose:** Simpler variant matching only the mean coordination distance
(not the full distribution).

**Additional arguments:**

| Key | Type | Description |
|-----|------|-------------|
| `target_distances` | list[float] | Target mean distances per pair (nm) |


### `coordination_tetrahedral`

**Purpose:** Matches both coordination distances and the tetrahedral order
parameter *q* for metal-centre coordination sites (e.g. zinc finger proteins).

**System-specific arguments:**

| Key | Type | Description |
|-----|------|-------------|
| `coord_sites` | list | Each entry: `[metal_idx, lig1_idx, lig2_idx, lig3_idx, lig4_idx]` (H5 indices) |
| `target_q` | float or list | Target tetrahedral order parameter (1.0 = perfect tetrahedron) |
| `target_site_distances` | list[float] | Target mean distance per (site, ligand_type) group |

**General arguments:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `q_weight` | float | `1.0` | Weight for *q* error |
| `dist_weight` | float | `1.0` | Weight for distance error |


### `coordination_tetrahedral_dist`

**Purpose:** Distribution-based variant of `coordination_tetrahedral`.
Matches per-group **distance distributions** and optionally the tetrahedral
order parameter $q$ **distribution** against reference data generated by
`diffmd-analyze`.  This is the recommended loss for optimising LJ parameters
of metal coordination sites (e.g. zinc fingers, cobalt rubredoxin).

**How it works:**
1. For each tetrahedral site, identifies the metal atom and its 4 ligand
   atoms by H5 atom indices
2. Groups ligands by atom *name* (from the H5 file) — sites with mixed
   ligand types produce one distance group per unique ligand type
3. For each frame (`vmap`), computes all metal–ligand distances with
   minimum-image PBC and the tetrahedral order parameter $q$
4. Builds KDE distributions across frames and MPI ranks for both distances
   (per group) and $q$ (per site)
5. Compares against reference distributions loaded from XVG files

**System-specific arguments** (set in `[nn.system_args.<name>]`):

| Key | Type | Description |
|-----|------|-------------|
| `coord_sites` | list[list[int]] | Each inner list: `[metal_idx, lig1_idx, lig2_idx, lig3_idx, lig4_idx]` — H5 **atom indices** (0-based) |
| `target_coord_dist` | string | Path to `.xvg` file with reference distance distributions (first column = bin centres, one column per ligand-type group) |
| `target_q_dist` | string | Path to `.xvg` file with reference $q$ distributions (first column = bin centres, one column per site) |
| `target_q` | float or list | Target mean $q$ value per site (default `1.0`) |

**General loss arguments** (set in `[nn.loss]`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `metric` | string | — | Error metric: `"mse"`, `"rmse"`, `"smape"`, `"l2e"` |
| `dist_weight` | float | `1.0` | Weight for distance-distribution KDE error |
| `q_weight` | float | `1.0` | Weight for mean $q$ error |
| `q_dist_weight` | float | `1.0` | Weight for $q$-distribution KDE error |
| `width_ratio` | float | `1.0` | Absolute KDE width in grid units: `width_ratio * bin_size` |
| `k_constraint` | float | `0.01` | Strength of harmonic/cubic parameter constraint |
| `constraint` | string | `None` | `"harmonic"` or `"cubic"` |
| `upper_boundary` / `boundary` | float | `None` | Soft upper bound on ε (sigmoid penalty) |
| `lower_boundary` | float | `None` | Soft lower bound on ε |

#### Generating reference distributions

Use `diffmd-analyze` (from the `gmx2hymd` package) on an all-atom
reference trajectory:

```bash
diffmd-analyze \
    -f trajectory.h5 \
    -q4 \
    --site 781 105 145 572 615 \
    --nbins 100 \
    --bw 0.05 \
    --dist-range 0.15 0.35 \
    --q-range 0.0 1.0 \
    -o training/
```

This produces two XVG files:
- `traj_comp_coord_dist.xvg` — distance distributions (columns: bins + one per ligand-type group)
- `traj_comp_q4_dist.xvg` — $q$ distributions (columns: bins + one per site)

#### XVG format

Both files use the GROMACS `.xvg` format.  Comment lines (`#`, `@`) are
skipped.  Data columns are whitespace-separated.  When loaded with
`np.loadtxt(file, comments=["#", "@"]).T`, the shape is
`(1 + n_columns, n_bins)`:

- Row 0: bin centres (e.g. distance in nm, or $q$ in [0, 1])
- Rows 1–N: probability densities, one per group/site

The number of data columns **must match** the number of ligand-type groups
(for distance) or sites (for $q$), otherwise the parser raises an error.

#### Working example: cobalt rubredoxin

One Co²⁺ metal centre coordinated by 4 cysteine-SG ligands (all same atom
type → 1 distance group, 1 $q$ distribution):

```toml
[nn]
systems = ["."]
n_epochs = 180
grad_method = "reverse"
equilibration = 1000
teacher_forcing = false

[nn.optimizer]
name = "adam"
learning_rate = 0.005

[nn.loss]
name = "coordination_tetrahedral_dist"
metric = "mse"
constraint = "harmonic"
k_constraint = 0.01
boundary = 50
dist_weight = 1.0
q_dist_weight = 1.0

# ─── IMPORTANT: system-specific args go in [nn.system_args."<system>"] ───
[nn.system_args."."]
# Tetrahedral site(s): [metal_idx, lig1, lig2, lig3, lig4]
# MUST be a list of lists, even for a single site!
coord_sites = [
    [781, 105, 145, 572, 615],
]

# Reference XVG files (relative to the system training directory)
target_coord_dist = "traj_comp_coord_dist.xvg"
target_q_dist = "traj_comp_q4_dist.xvg"

[nn.model]
train_sigma = false
LJ_type_param = [
    ['SZ', 0.366359, 1.046000, "on_sigma", "on_eps"],
    ['Zn', 0.226466, 0.013820],
    # ... other atom types (fixed) ...
]
```

#### Common pitfalls

1. **`[nn.system_args."."]` must NOT be commented out.**  If this section
   header is commented, all keys below it (coord_sites, target_coord_dist,
   target_q_dist) fall into `[nn.loss]`, causing
   `KeyError: 'system_args'` at startup.

2. **`coord_sites` must be a list of lists**, e.g. `[[781, 105, 145, 572, 615]]`.
   A flat list `[781, 105, 145, 572, 615]` (missing the inner brackets)
   causes a `TypeError` because the parser iterates over sites and calls
   `len()` on each element.

3. **`target_coord_dist` and `target_q_dist` belong in `[nn.system_args."."]`**,
   NOT in `[nn.loss]`.  The parser looks for them in the system-args section;
   if they are in the loss section they are silently ignored and the
   reference distributions are never loaded.

4. **`coord_pairs` is NOT used** by `coordination_tetrahedral_dist` (only by
   `coordination_distance` / `coordination_distance_dist`).  Including it
   in system_args causes `TypeError: unexpected keyword argument`.

5. **Number of XVG data columns must match.**  With 1 site and 4 same-type
   ligands → 1 distance group → the coord_dist XVG needs exactly 1 data
   column (+ 1 bin column = 2 total).  With 2 sites → 2 $q$ columns, etc.


### Metrics

All loss functions accept a `metric` parameter defining the error measure:

| Name | Formula | Notes |
|------|---------|-------|
| `mse` | `mean((pred - target)²)` | Default, well-behaved gradients |
| `rmse` | `sqrt(mean((pred - target)²))` | Penalizes large deviations more |
| `smape` | `mean(\|pred - target\| / (\|pred\| + \|target\|))` | Scale-independent, range [0, 1] |
| `l2e` | `\|\|pred - target\|\|₂` | Euclidean distance |
| `wasserstein_1d` | 1-D Wasserstein (earth-mover) distance | Distribution-aware; well suited to KDE/histogram targets |


### Boundary Constraint

The soft boundary penalty uses a sigmoid function:

$$
P(\epsilon) = \frac{C}{2} \sum_i \sigma\bigl((\epsilon_i - B_\text{upper}) \cdot S\bigr) + \frac{C}{2} \sum_i \sigma\bigl((B_\text{lower} - \epsilon_i) \cdot S\bigr)
$$

where $C$ = `boundary_C` (amplitude), $S$ = `boundary_S` (steepness),
and $B$ is the boundary value.  The penalty is near zero when parameters
are far from the boundary and ramps up smoothly as they approach it.

**Important:** If neither `upper_boundary`/`boundary` nor `lower_boundary`
is set, **no boundary penalty is applied** and parameters are only
constrained by the hard clipping in the optimizer (see below).


### Hard Clipping (Optimizer Projection)

After each gradient update, parameters are projected into a bounding box:

| Key | Default | Description |
|-----|---------|-------------|
| `clip_sigma_min` | `0.05` nm | Hard minimum for sigma parameters |
| `clip_sigma_max` | `2.0` nm | Hard maximum for sigma parameters |
| `clip_epsilon_min` | `0.001` kJ/mol | Hard minimum for epsilon parameters |
| `clip_epsilon_max` | `100.0` kJ/mol | Hard maximum for epsilon parameters |

These are set in the `[nn]` section (not `[nn.loss]`).  The `LJ_param`
array is packed as `[sigma_0, ..., sigma_n, epsilon_0, ..., epsilon_m]`,
so the bounds are applied per-type accordingly.



## References

1. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M.
   (2018). Automatic Differentiation in Machine Learning: a Survey.
   *Journal of Machine Learning Research*, 18(153), 1–43.
   — §3.1: forward and reverse mode produce identical derivatives.

2. Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles
   and Techniques of Algorithmic Differentiation*. 2nd ed. SIAM.
   — Theorem 3.1: JVP is the transpose of VJP.

3. Häfner, D. & Vicentini, F. (2021). mpi4jax: Zero-copy MPI communication
   of JAX arrays. *Journal of Open Source Software*, 6(65), 3419.
   — JVP rules for MPI collective primitives.

4. Bradbury, J. et al. (2018). JAX: composable transformations of
   Python+NumPy programs. https://github.com/jax-ml/jax
   — `jax.jvp`, `jax.value_and_grad`, `jax.checkpoint` documentation.

> **Note on memory:** With reverse-mode, `n_print` controls how many
> frames are stored in the carry stack.  `n_print=10` → ~400 MB transient
> peak; `n_print=400` → ~16 GB.  Tune this value when using `grad_method = "reverse"`.


---

## Optimizer Guide

∂-aMD uses [optax](https://optax.readthedocs.io/en/latest/api.html)
for gradient-based optimization.  The optimizer is set in `[nn.optimizer]`
of the training TOML.  This section compares the most useful optimizers
for LJ parameter fitting and gives practical recommendations.

### Available Optimizers

| Name (TOML `name`) | Type | Key Hyperparams | Best For |
|---------------------|------|-----------------|----------|
| `adam` | Adaptive | `learning_rate`, `b1`, `b2` | General-purpose default |
| `adamw` | Adaptive + decay | `learning_rate`, `b1`, `b2`, `weight_decay` | Preventing parameter drift |
| `adabelief` | Adaptive | `learning_rate`, `b1`, `b2` | Noisy/stochastic losses |
| `sgd` | Fixed step | `learning_rate` | Fine-tuning near convergence |
| `rmsprop` | Adaptive | `learning_rate`, `decay` | Non-stationary objectives |
| `lion` | Sign-based | `learning_rate`, `b1`, `b2` | Memory-efficient, large batch |
| `lamb` | Adaptive + layerwise scaling | `learning_rate`, `b1`, `b2`, `weight_decay` | Multi-system training |
| `noisy_sgd` | Stochastic | `learning_rate`, `eta` | Escaping local minima |

Any optimizer listed in the
[optax API](https://optax.readthedocs.io/en/latest/api/optimizers.html)
can be used — just set `name` to the optax function name and pass its
keyword arguments as TOML keys.

### Recommended Configurations

**Smooth convergence (CG lipids, ≤20 params):**
```toml
[nn.optimizer]
name = "adam"
learning_rate = 0.005
b1 = 0.9
b2 = 0.999
```
Adam's moving-average momentum smooths out the stochastic noise from
short MD trajectories.  `b1 = 0.9` is a safe default.

**Aggressive start, then refine (when loss plateaus):**
```toml
[nn.optimizer]
name = "adam"
b1 = 0.9
b2 = 0.999

[nn.optimizer.learning_rate]
schedule = "exponential_decay"
init_value = 0.01
transition_steps = 100
decay_rate = 0.95
transition_begin = 20
```
Starts at `lr=0.01`, holds for 20 epochs, then decays by 5% every 100
steps.  Good for long runs (>200 epochs) where you want fast initial
progress and stable convergence.

**Noisy or multi-system losses:**
```toml
[nn.optimizer]
name = "adabelief"
learning_rate = 0.0002
b1 = 0.1
b2 = 0.4
```
AdaBelief adapts the step size based on the *belief* in the gradient
direction — when gradients are noisy (e.g. multiple systems with
different characteristics), it takes smaller, more cautious steps.
Lower `b1`/`b2` values discount old gradient history faster, which helps
when the loss landscape shifts between systems.

**Gradient clipping + optimizer chaining:**
```toml
chain = true

[[nn.optimizer]]
name = "clip_by_global_norm"
max_norm = 10.0

[[nn.optimizer]]
name = "adam"
learning_rate = 0.005
```
Chains a gradient clipping transform before Adam.  This prevents
gradient explosions from rare bad trajectories (e.g. atom overlap after
restart).  The `chain = true` flag and **double brackets `[[...]]`** are
required for chaining.

**Fine-tuning near convergence:**
```toml
[nn.optimizer]
name = "sgd"
learning_rate = 0.0001
```
When the loss is already low, switching to plain SGD with a tiny learning
rate avoids the momentum overshoot of Adam.

### Learning Rate Schedules

Any optax schedule can be used by making `learning_rate` a sub-table:

```toml
[nn.optimizer.learning_rate]
schedule = "cosine_decay_schedule"   # optax schedule function name
init_value = 0.01
decay_steps = 500
alpha = 0.0001                        # final LR
```

Other useful schedules:
- `exponential_decay` — multiplicative decay every N steps
- `piecewise_constant_schedule` — manual step-wise LR drops
- `warmup_cosine_decay_schedule` — ramp-up then cosine decay
- `linear_schedule` — linearly interpolate from init to end value

### Training TOML Reference (full `[nn]` section)

Below is an annotated example covering all optimizer-related keys
(from `atomistic_protein/PROVA/dopc_test/training/training.toml`):

```toml
[nn]
systems = ["DOPC"]              # directories containing system data
n_epochs = 200                  # total training iterations
grad_method = "jvp"             # "reverse", "jvp", or "finite_diff"
fd_epsilon = 1e-4               # FD step size (only for finite_diff)
double_precision = false        # enable float64 (recommended for PME/NVE)

equilibration = 50000           # non-differentiable warmup steps per epoch
teacher_forcing = false         # true = continue from last trajectory frame
shuffle = true                  # randomize system order each epoch
batch_size = 1                  # accumulate gradients over N steps before update

# Clipping bounds applied AFTER each gradient update
clip_sigma_min = 0.05           # minimum σ (nm)
clip_sigma_max = 2.0            # maximum σ (nm)
clip_epsilon_min = 0.001        # minimum ε (kJ/mol)
clip_epsilon_max = 100.0        # maximum ε (kJ/mol)

# Periodic longer equilibration (optional)
# n_epochs_longer = 5            # every N epochs…
# n_steps_longer = 10000         # …run this many warmup steps instead

[nn.optimizer]
name = "adabelief"
learning_rate = 0.0002
b1 = 0.1
b2 = 0.4

[nn.loss]
name = "density_and_apl"
metric = "mse"
width_ratio = 0.1
density_weight = 1
apl_weight = 1
boundary = 20                   # soft upper bound on ε (deprecated: use upper_boundary)
boundary_S = 2                  # steepness of boundary sigmoid
# constraint = "harmonic"       # restrain params toward initial values
# k_constraint = 0.1            # strength of harmonic constraint
```

### Best Practices

1. **Start with Adam, `lr = 0.005`.**  It works for most systems
   out of the box.  Only switch after you see a specific problem.

2. **Use gradient clipping** (`chain = true` + `clip_by_global_norm`)
   for atomistic systems or when you see NaN/Inf losses.

3. **Reduce `learning_rate` if the loss oscillates** wildly between epochs.
   Increase it if convergence is too slow.

4. **Lower `b1`** (e.g. 0.1–0.5) for very noisy losses or when training
   multiple systems simultaneously.

5. **Use `batch_size > 1`** to accumulate gradients over several MD steps
   before updating, which reduces noise at the cost of slower updates.

6. **Use LR scheduling** for long runs (>100 epochs): start high, decay
   exponentially.  This is almost always better than a fixed LR.

7. **Harmonic constraint** (`constraint = "harmonic"`, `k_constraint = 0.01`)
   prevents parameters from drifting far from their initial (force-field)
   values.  Useful when you trust the starting FF and want small corrections.

8. **Boundary penalty** (soft sigmoid via `upper_boundary`/`boundary_S`)
   is gentler than hard clipping and provides gradient signal to push
   parameters back inward.  Combine both for robustness.

9. **Double precision** (`double_precision = true` or `--double-precision`)
   is recommended for atomistic PME systems to avoid gradient explosion
   from float32 Ewald sums.  For CG systems, float32 is fine.


---

## Restarting an MPI Optimization

### How Checkpoints Work

At the end of each epoch, ∂-aMD writes a checkpoint directory:
```
output_dir/step_<epoch>/cpt/
```
The checkpoint (v3 format) contains:
- `epoch` — the next epoch to run
- `params` — current LJ parameters
- `state` — optimizer state (Adam moments, etc.)
- `keys` — per-rank PRNG keys (shape `[n_ranks, ...]`)
- `positions`, `velocities`, `box_sizes` — per-rank, per-system arrays
- `init_temps` — which systems have been thermalized
- `start_positions`, `start_velocities`, `start_box_sizes` — equilibration restart state (if applicable)

### Restarting a Stopped Job

Suppose your optimization ran for 72 epochs on 4 MPI ranks and was killed.
The last complete checkpoint is in `output_dir/step_71/cpt/`.  To restart:

```bash
# Same number of ranks as the original run
mpirun -np 4 python -m diff_md optimize \
    --model training.toml \
    --destdir output_dir \
    --restart output_dir/step_71/cpt
```

The run will resume from **epoch 72** with:
- Each rank's positions, velocities, and box restored to their **own** state
  (not rank 0's — this is the v3 replica-safe checkpoint format)
- Optimizer momentum/variance terms intact (no cold-restart transient)
- PRNG streams continued from where they left off

**Important:** You must use the **same number of MPI ranks** as the original
run, since each checkpoint slot is indexed by rank.

### Checking What Epoch a Checkpoint Contains

```bash
ls output_dir/step_*/cpt/ | tail -1
# → output_dir/step_71/cpt/
# The checkpoint stores epoch=72 (the NEXT epoch to run)
```

Or inspect programmatically:
```python
import orbax.checkpoint
ckpt = orbax.checkpoint.PyTreeCheckpointer()
restored = ckpt.restore("output_dir/step_71/cpt", item={"epoch": 0})
print(restored["epoch"])  # → 72
```

### Changing `n_epochs` on Restart

The training loop runs `range(start_epoch, start_epoch + n_epochs)`.
If the original TOML had `n_epochs = 200` and you restart at epoch 72,
it will run epochs 72–271 (200 additional epochs).  To run only until
epoch 200 total, set `n_epochs = 128` in the TOML before restarting.

### Backward Compatibility

- **v3 checkpoints** (current): per-rank arrays, key `"keys"` (plural).
  Requires the same number of ranks.
- **v2 checkpoints** (old): single-rank state, key `"key"` (singular).
  Restored with a warning — all ranks start from rank 0's state.
- **v1 checkpoints** (legacy): only `epoch`, `params`, `opt_state`.
  Positions/velocities reset to the initial H5 file.


---

## Why Multiple MPI Replicas Accelerate Convergence

### The Gradient Averaging Effect

Each MPI rank runs an **independent MD trajectory** from a different
configuration (different initial random seed → different thermal noise).
The loss and its gradient are computed locally, then **averaged across
ranks** via `allreduce(SUM)`:

$$
\nabla_\theta L_\text{global} = \frac{1}{N_\text{ranks}} \sum_{r=1}^{N_\text{ranks}} \nabla_\theta L_r(\theta)
$$

This is mathematically equivalent to computing the gradient from a single
trajectory $N_\text{ranks}$ times longer — but runs in parallel, so
wall-clock time stays the same.

### Variance Reduction

A single short MD trajectory produces a noisy gradient because the
sampled configurations are correlated (the system explores a small
region of phase space).  By averaging $N$ independent trajectories:

$$
\text{Var}(\nabla L_\text{global}) = \frac{1}{N} \text{Var}(\nabla L_\text{single})
$$

The gradient noise drops by $1/N$.  This means:
- **Smoother loss curves** — fewer wild oscillations between epochs
- **Larger stable learning rate** — less noise → Adam/SGD can take
  bigger steps without overshooting
- **Better convergence** — the optimizer follows the true gradient
  direction more closely instead of random-walking

### Practical Scaling

| Ranks | Gradient noise | LR headroom | Convergence |
|-------|---------------|-------------|-------------|
| 1 | High | Must use small LR | Slow, noisy |
| 2 | 0.71× | ~1.4× higher LR safe | Noticeably smoother |
| 4 | 0.50× | ~2× higher LR safe | Good balance |
| 8 | 0.35× | ~3× higher LR safe | Near-deterministic |

**Rule of thumb:** Use as many ranks as you have GPUs.  Beyond ~8 ranks
the returns diminish (each trajectory is already long enough to sample
the relevant phase space), and communication overhead starts to matter.

### Example: 4-rank Zinc Protein Optimization

```bash
# Each rank gets its own GPU
mpirun -np 4 \
    --map-by slot:PE=1 \
    --bind-to core \
    bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; \
             python -m diff_md optimize \
             --model training.toml \
             --destdir output_4rank'
```

At 4 ranks the gradient estimate is 2× less noisy than a single rank,
allowing you to use `learning_rate = 0.01` instead of `0.005` and reach
the same loss in roughly half the number of epochs.


## Metrics

Metrics define how the simulated observable (e.g. a KDE distribution) is
compared with the reference data.  They are specified by the `metric` key
in the `[nn.loss]` section of the training TOML:

```toml
[nn.loss]
name = "radius_of_gyration_dist"
metric = "Kullback_Leibler"        # or "mse", "wasserstein_1d", etc.
```

All metrics follow the same call convention:
```python
result = metric(predictions, targets, axis=None)
```
where `predictions` and `targets` are JAX arrays (possibly with different
ranks — broadcasting is handled internally for distribution metrics).
All metrics are JIT-compiled and differentiable via `jax.grad`.

### Available metrics

| Metric | Formula | Gradient-safe | Symmetric | Best for |
|--------|---------|:---:|:---:|----------|
| `mse` | $\frac{1}{N}\sum(p_i - t_i)^2$ | Yes | Yes | General purpose |
| `rmse` | $\sqrt{\text{MSE}}$ | Yes | Yes | Same units as data |
| `smape` | $\frac{1}{N}\sum\frac{|p_i-t_i|}{|p_i|+|t_i|}$ | Yes | Yes | Scale-invariant |
| `l2e` | $\|p - t\|_2$ | Yes | Yes | Euclidean distance |
| `Kullback_Leibler` | $\sum p_i \log(p_i / q_i)$ | Yes\* | No | Distribution matching |
| `wasserstein_1d` | $\sum|\text{CDF}_P - \text{CDF}_Q|$ | Yes | Yes | Distribution matching |

\* With `eps = 1e-8` floor to bound gradients.

### `mse`

Mean Squared Error.  The default workhorse metric for matching scalar
observables (mean Rg, mean distances) and distribution profiles.

$$
\text{MSE}(p, t) = \frac{1}{N} \sum_{i=1}^N (p_i - t_i)^2
$$

**Properties:** Symmetric, differentiable everywhere, gradients scale
linearly with the error.  Penalises large deviations quadratically.

### `rmse`

Root Mean Squared Error.  Same as MSE but with a square root, so the
result has the same units as the data (e.g. nm for distances).

$$
\text{RMSE}(p, t) = \sqrt{\frac{1}{N} \sum_{i=1}^N (p_i - t_i)^2}
$$

### `smape`

Symmetric Mean Absolute Percentage Error.  Scale-invariant — useful when
comparing distributions with very different magnitudes across bins.

$$
\text{SMAPE}(p, t) = \frac{1}{N} \sum_{i=1}^N \frac{|p_i - t_i|}{|p_i| + |t_i|}
$$

Bins where both `p` and `t` are zero contribute 0 (safe division).

### `l2e`

L2 (Euclidean) Error — the vector norm of the difference.

$$
\text{L2E}(p, t) = \|p - t\|_2 = \sqrt{\sum_{i=1}^N (p_i - t_i)^2}
$$

Unlike MSE, L2E does not divide by `N`, so it is sensitive to the number
of bins.

### `Kullback_Leibler`

Forward Kullback-Leibler divergence: $KL(\text{target} \| \text{sim})$.
Both inputs are normalised internally so that they sum to 1.

$$
KL(p \| q) = \sum_{i} p_i \log \frac{p_i}{q_i}
$$

where $p$ = target (reference distribution), $q$ = predictions (simulated
KDE).  Bins where $p \approx 0$ are excluded (zero contribution).

**Numerical safety:**
- Uses `eps = 1e-8` in the log domain.  This bounds the per-bin gradient
  to $|\partial KL / \partial q_i| \leq 1/\varepsilon = 10^8$, preventing
  NaN during backpropagation when the simulated distribution has near-zero
  bins that the target covers.
- The normalisation denominator uses a separate `1e-30` floor to avoid
  division by zero for all-zero inputs.

**Properties:**
- **Not symmetric:** $KL(p\|q) \neq KL(q\|p)$ in general.
- **Mode-covering:** Heavily penalises regions where the target has density
  but the simulation does not ("zero-avoiding" behavior).  This drives
  the simulation to cover the full support of the target.
- **Gradients:** Through the fixed-width Gaussian KDE and the simulation — fully
  differentiable.  The `eps` floor prevents gradient explosion.

**When to use:** Distribution-matching losses (`radius_of_gyration_dist`,
`coordination_distance_dist`) where you want the simulated KDE to cover
all features of the reference distribution.  Especially useful for
multi-modal distributions.

**Example:**
```toml
[nn.loss]
name = "radius_of_gyration_dist"
metric = "Kullback_Leibler"
constraint = "harmonic"
k_constraint = 1.0

[nn.system_args.my_system]
target_dist = "target_pdf.npy"    # shape (2, n_bins): row 0 = bin centres, row 1 = PDF
resname = "PCP"
n_chains = 1
width_ratio = 3                   # KDE bandwidth factor (1-5 recommended)
                                  # => absolute width = 3 * bin_size
```

**Tip:** If you see loss values ~50-70 nats, this means the simulated and
target distributions barely overlap.  Consider:
1. Increasing `width_ratio` to broaden the KDE (try 3-5)
2. Using `wasserstein_1d` instead (geometrically meaningful even without
   overlap)
3. Setting `max_grad_norm` in `[nn]` to clip extreme gradients

### `wasserstein_1d`

1-D Wasserstein distance (Earth Mover's Distance) via the CDF trick.
Both inputs are normalised internally to sum to 1.

$$
W_1(p, q) = \sum_{i} |\text{CDF}_p(i) - \text{CDF}_q(i)|
$$

where $\text{CDF}_p(i) = \sum_{j \leq i} p_j$ is the cumulative
distribution function.  The bin width is a constant scale factor that
does not affect optimisation.

**Properties:**
- **Symmetric:** $W_1(p, q) = W_1(q, p)$.
- **True metric:** Satisfies the triangle inequality.
- **Geometrically meaningful:** Measures the "work" needed to transport
  one distribution to the other.  Provides useful gradients even when
  the distributions have non-overlapping supports (unlike KL which
  produces infinite divergence).
- **No log operation:** Inherently numerically safe — no risk of
  log(0) or exploding gradients.
- **Smooth gradients:** The absolute value in $|\text{CDF}_p - \text{CDF}_q|$
  has subgradient 0 at the crossing point, but JAX handles this correctly.

**When to use:** As an alternative or companion to `Kullback_Leibler` for
distribution matching.  Especially recommended when:
- The initial parameters produce a simulation distribution far from the
  target (KL divergence would be very large with weak gradients)
- You want symmetric, geometrically meaningful distance
- Gradient stability is a priority

**Example:**
```toml
[nn.loss]
name = "radius_of_gyration_dist"
metric = "wasserstein_1d"
constraint = "harmonic"
k_constraint = 1.0
```

### Choosing a Metric for Distribution Matching

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Distributions overlap well | `Kullback_Leibler` | Strong signal in overlapping region |
| Distributions barely overlap | `wasserstein_1d` | Meaningful gradient even without overlap |
| Unknown overlap | `wasserstein_1d` | Safer default, no NaN risk |
| Multi-modal target | `Kullback_Leibler` | Mode-covering behavior |
| Simple scalar matching | `mse` | Standard, well-understood |

## KDE for Distribution Losses

All distribution-based losses in $\partial$-aMD use a fixed-width Gaussian KDE
to transform discrete samples into a differentiable PDF.  The smoothing width is

$$
h = \text{width\_ratio} \times \text{bin\_size}
$$

where `bin_size` is the spacing between evaluation points in the reference grid.
This is an **absolute** kernel width.  It is not passed through
`gaussian_kde(bw_method=...)`, so it does not get multiplied by the sampled
trajectory standard deviation.

This change matters for narrow simulated distributions: under the old adaptive
interpretation, a small sample variance could collapse the replayed KDE into an
ultra-sharp spike and produce pathological KL gradients even when `width_ratio`
looked reasonable in the TOML.

### Tuning Guidelines

| `width_ratio` | Effect | Use Case |
|---------------|--------|----------|
| `0.1 – 0.5` | Sharp KDE, preserves narrow peaks | High-resolution structural features |
| `1.0` | Balanced default, one kernel width per bin | General-purpose starting point |
| `1.5 – 3.0` | Smoother KDE, lower gradient noise | Limited sample counts or noisy trajectories |
| `3.0 – 5.0` | Very smooth KDE, long-range gradient signal | Early training when PDFs barely overlap |

### Best Practices

1. Match `width_ratio` to the smoothing used to generate the target reference distribution.
2. Increase `width_ratio` if the replayed PDF is too spiky or KL gradients are unstable.
3. Decrease `width_ratio` if narrow physical peaks are being over-smoothed.
4. Use `wasserstein_1d` when the replayed and target PDFs start far apart; it remains informative even with weak overlap.

### Diagnostic Tool

Use [tools/compare_rg_distributions.py](tools/compare_rg_distributions.py) to compare the target PDF, saved optimization PDFs, and a replayed PDF from any H5 trajectory:

```bash
python tools/compare_rg_distributions.py \
   --trajectory roba/polimero/distrib/simulation_pt2.h5 \
   --target roba/polimero/distrib/target_pdf.npy \
   --step0 roba/polimero/distrib/dist_saga/dist_saga/distr_test_4_adam/step_0/pmetac-15_005_Rg.npy \
   --step0v2 roba/polimero/distrib/dist_saga/dist_saga/distr_test_4_adam/step_0/pmetac-15_005_Rg_step0v2.npy \
   --resname PCP --n-chains 1 --width-ratio 3 \
   --destdir roba/polimero/distrib/rg_compare_out --legacy-adaptive
```

The tool writes `comparison.png` and `comparison.json` so it is easy to check
whether a trajectory replay is consistent with the saved training artifacts.

  Sharp edge (documented, not fixed): the optional constraint(model.LJ_param, k, param_constraints) regularizer at losses.py:592 operates on the full flat LJ_param
  symmetrically on all ranks; after allreduce(SUM) it multiplies private-row constraint gradients by world_size. v1 workaround: user adjusts k_constraint. Follow-up:
  gate the constraint to "rows used by this rank" before backward.

  To run replicas-of-same-system: add replica_of = "name" to two or more [nn.system_args.<dir>] entries; the topology equality check will
  guard against accidental mismatches.

#ATTENTO!!!
Cambio in SPME
The main root cause was PME, not the TOML setup: pme_order, mesh, cutoff, and sigma="auto" were already matching the GROMACS-style setup. Diff-MD’s reciprocal filter was using a continuous sinc^(-2p) B-spline deconvolution, while GROMACS/SPME uses the discrete cardinal B-spline modulus. I changed that in nonbonded.py:422-456, so filter_density() now applies the GROMACS-compatible SPME deconvolution

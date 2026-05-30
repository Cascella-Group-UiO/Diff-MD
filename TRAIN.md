# Diff-aMD Training Guide

## How LJ Parameters Are Specified in `training.toml`

Diff-MD supports two modes for specifying which Lennard-Jones parameters to optimize.
Both are defined inside the `[nn.model]` section of `training.toml`.

---

### Mode 1: Pair Mode (`LJ_param`)

Directly specify sigma and epsilon for each **pair** of atom types.
Both sigma and epsilon can be independently made trainable per pair.

```toml
[nn.model]
LJ_param = [
    # [type_A,  type_B,  sigma,    epsilon,  flags...]
    ["SZ",     "ZN",    2.50e-01, 0.500,    "on_eps"],              # train epsilon
    ["NZ",     "ZN",    2.60e-01, 0.400,    "on_sigma", "on_eps"],  # train both
    ["SZ",     "NZ",    3.00e-01, 0.300,    "on_eps", "cs"],        # train eps + constrain
    ["SZ",     "SZ",    3.10e-01, 0.250,    "on_sigma"],            # train sigma only
    ["NZ",     "NZ",    3.20e-01, 0.350],                           # fixed
    ["ZN",     "ZN",    2.00e-01, 0.100],                           # fixed
]
```

**Format:** `[type_A, type_B, sigma, epsilon, ...flags]`

| Flag | Effect |
|------|--------|
| `"on_eps"` or `"on_epsilon"` | Train **epsilon** for this pair |
| `"on_sigma"` or `"on_sgm"` | Train **sigma** for this pair |
| `"cs"` or `"cs_eps"` or `"cs_epsilon"` | Constrain epsilon toward initial value |
| `"cs_sigma"` or `"cs_sgm"` | Constrain sigma toward initial value |
| _(none)_ | Fixed — not optimized |

**Behavior:**
- If **any** row has explicit train flags → only flagged params are trainable (selective).
- If **no** row has train flags → all epsilons are trainable by default.
- The trainable parameter vector is packed as `[sigma_values..., epsilon_values...]`.
- Pairs not listed here keep their force-field defaults.

---

### Mode 2: Type Mode (`LJ_type_param`)

Specify sigma and epsilon **per atom type**. Cross-interactions are built automatically
via combining rules (arithmetic for sigma, geometric for epsilon).
Both sigma and epsilon can be independently made trainable.

```toml
[nn.model]
LJ_type_param = [
    # [type,  sigma,    epsilon,  flags...]
    ["ZN",   2.00e-01, 0.100,    "on_eps", "on_sigma"],        # train eps + sigma
    ["SZ",   3.10e-01, 0.250,    "on_eps"],                    # train eps only
    ["NZ",   3.20e-01, 0.350,    "on_sigma"],                 # train sigma only
    ["CT",   3.40e-01, 0.457],                                # fixed
    ["N",    3.25e-01, 0.711],                                # fixed
    ["O",    2.96e-01, 0.879],                                # fixed
]
```

**Format:** `[type_name, sigma, epsilon, ...flags]`

| Flag | Effect |
|------|--------|
| `"on_eps"` or `"on_epsilon"` | Train **epsilon** for this type |
| `"on_sigma"` or `"on_sgm"` | Train **sigma** for this type |
| `"cs"` or `"cs_eps"` or `"cs_epsilon"` | Constrain epsilon toward initial value |
| `"cs_sigma"` or `"cs_sgm"` | Constrain sigma toward initial value |
| _(none)_ | Fixed |

**Behavior:**
- If **any** row has explicit train flags → only flagged params are trainable.
- If **no** row has train flags → all epsilons are trainable; sigma is controlled
  by the global switch `train_sigma = true/false` (default `false`).
- The trainable parameter vector `LJ_param` is packed as `[sigma_values..., epsilon_values...]`.
- Cross-interactions use combining rules from the `options.toml`:
  - Sigma: arithmetic mean `(σ_i + σ_j) / 2` or geometric `sqrt(σ_i * σ_j)`
  - Epsilon: always geometric `sqrt(ε_i * ε_j)`

---

### How the Internal Mapping Works

1. Each unique type name gets a 0-based index (`name_to_type` dict).
2. In **pair mode**: `lj_sigma_idx` and `lj_epsilon_idx` record which pair flat-indices
   are trainable. During forward pass, config's sigma/epsilon tables are overridden
   at the flagged pair positions with the current trained values.
3. In **type mode**: `lj_sigma_idx` and `lj_epsilon_idx` record which type indices
   are trainable. `lj_sigma_ref` / `lj_epsilon_ref` hold the full reference arrays.
   During forward pass, the trained values overwrite the reference at the flagged indices,
   then combining rules build the full `(n_types, n_types)` sigma and epsilon tables.
4. The optimizer sees **only** `GeneralModel.LJ_param` as a JAX leaf (pytree node).
   All other fields are static metadata (`pytree_node=False`).

---

## Concrete Examples for Zn Coordination (SZ, NZ, ZN)

### Example A: Optimize SZ–ZN and NZ–ZN epsilon separately (pair mode)

Only the two coordination pairs have epsilon trainable; everything else is fixed.

```toml
[nn.model]
LJ_param = [
    ["SZ",  "ZN",  2.50e-01, 0.500, "on_eps"],   # trainable: SZ-ZN epsilon
    ["NZ",  "ZN",  2.60e-01, 0.400, "on_eps"],   # trainable: NZ-ZN epsilon
    ["SZ",  "SZ",  3.10e-01, 0.250],          # fixed
    ["SZ",  "NZ",  3.00e-01, 0.300],          # fixed
    ["NZ",  "NZ",  3.20e-01, 0.350],          # fixed
    ["ZN",  "ZN",  2.00e-01, 0.100],          # fixed
]
```

Trainable vector: `[eps_SZ-ZN, eps_NZ-ZN]` (2 values).

### Example A2: Optimize sigma AND epsilon for SZ–ZN pair (pair mode)

```toml
[nn.model]
LJ_param = [
    ["SZ",  "ZN",  2.50e-01, 0.500, "on_sigma", "on_eps"],  # both trainable
    ["NZ",  "ZN",  2.60e-01, 0.400],          # fixed
    ["SZ",  "SZ",  3.10e-01, 0.250],          # fixed
    ["NZ",  "NZ",  3.20e-01, 0.350],          # fixed
    ["ZN",  "ZN",  2.00e-01, 0.100],          # fixed
]
```

Trainable vector: `[sigma_SZ-ZN, eps_SZ-ZN]` (2 values).

### Example B: Optimize sigma AND epsilon of ZN, NZ, SZ together (type mode)

All three types have both sigma and epsilon trainable. Other types stay fixed.

```toml
[nn.model]
LJ_type_param = [
    ["ZN",  2.00e-01, 0.100, "on_eps", "on_sigma"],   # train eps + sigma
    ["SZ",  3.10e-01, 0.250, "on_eps", "on_sigma"],   # train eps + sigma
    ["NZ",  3.20e-01, 0.350, "on_eps", "on_sigma"],   # train eps + sigma
    ["CT",  3.40e-01, 0.457],                       # fixed
    ["N",   3.25e-01, 0.711],                       # fixed
    ["O",   2.96e-01, 0.879],                       # fixed
]
```

Trainable vector: `[sigma_ZN, sigma_SZ, sigma_NZ, eps_ZN, eps_SZ, eps_NZ]` (6 values).

### Example C: Optimize only epsilon of ZN (type mode, minimal)

```toml
[nn.model]
LJ_type_param = [
    ["ZN",  2.00e-01, 0.100, "on_eps"],   # train epsilon only
    ["SZ",  3.10e-01, 0.250],          # fixed
    ["NZ",  3.20e-01, 0.350],          # fixed
    ["CT",  3.40e-01, 0.457],          # fixed
]
```

Trainable vector: `[eps_ZN]` (1 value).

### Example D: Optimize sigma of SZ and NZ, epsilon of ZN (type mode, mixed)

```toml
[nn.model]
LJ_type_param = [
    ["ZN",  2.00e-01, 0.100, "on_eps"],                        # train eps
    ["SZ",  3.10e-01, 0.250, "on_sigma"],                  # train sigma
    ["NZ",  3.20e-01, 0.350, "on_sigma", "cs_sigma"],     # train sigma + constrain it
    ["CT",  3.40e-01, 0.457],                               # fixed
]
```

Trainable vector: `[sigma_SZ, sigma_NZ, eps_ZN]` (3 values).

---

## Loss Function Reference

Loss functions live in `src/diff_md/losses.py`. Each loss function has the same signature pattern:

```python
def loss_name(
    model, system, key, start_temperature, comm,  # always provided by optimize.py
    <system_args>,                                 # from [nn.system_args.<name>]
    <loss_args>,                                   # from [nn.loss]
):
    # 1. Extract LJ tables from model
    sgm, epsl, constraints, types = get_LJ_param(model, system.config, system.types)
    # 2. Run simulation
    trj, key, config = simulator(model, system.positions, ..., sgm, epsl, ...)
    # 3. Compute observable from trajectory
    # 4. Compare to target → error
    # 5. Add constraints/boundaries
    return error, (output_dict, trj, key, config, types)
```

### Available Losses

| Name | Target Observable | System Args |
|------|-------------------|-------------|
| `density_and_apl` | Lateral density profile + area per lipid | `z_range`, `com_type`, `n_lipids`, `target_density`, `target_apl` |
| `radius_of_gyration` | Mean Rg | `n_chains`, `chain_indices`, `chain_masses`, `target_rg` |
| `radius_of_gyration_dist` | Rg probability distribution (KDE) | `n_chains`, `chain_indices`, `chain_masses`, `data_range`, `target_dist` |
| `radius_of_gyration_median` | Median Rg across replicas | same as `radius_of_gyration` |
| `radius_of_gyration_filter_repls` | Filtered mean Rg | same as `radius_of_gyration` |
| `radius_of_gyration_and_end_to_end` | Rg + end-to-end distance | same + `target_end_to_end` |

### Available Metrics

Specified in `[nn.loss] metric = "..."`:

| Metric | Description |
|--------|-------------|
| `mse`  | Mean squared error |
| `rmse` | Root mean squared error |
| `smape`| Symmetric mean absolute percentage error |
| `l2e`  | L2 norm of error |

### Constraints and Boundaries

Diff-MD provides two complementary mechanisms for keeping LJ parameters
in a physically meaningful range: **soft** loss penalties that gradually
push parameters back, and **hard** clipping that enforces strict bounds
after every optimizer step.

#### 1. Soft Boundaries (Loss Penalties)

Add smooth penalty terms to the loss when parameters approach boundaries.
Use these to guide the optimizer away from unphysical regions without
creating sharp gradient discontinuities.

```toml
[nn.loss]
constraint = "harmonic"   # or "cubic" — penalty for deviating from initial values
k_constraint = 0.01       # strength of the constraint penalty

# Soft sigmoid boundaries for parameter values
upper_boundary = 50.0     # penalty when epsilon > this value
lower_boundary = 0.05     # penalty when epsilon < this value
boundary_S = 2            # sigmoid steepness (higher = sharper transition)
boundary_C = 500          # sigmoid amplitude (higher = stronger penalty)
```

How the sigmoid boundary works:  when a parameter *p* exceeds
`upper_boundary`, a term `boundary_C * σ(boundary_S * (p − upper))` is
added to the loss; the analogous term fires below `lower_boundary`.
Increasing `boundary_S` makes the onset zone narrower (approaching a hard
wall); increasing `boundary_C` raises the penalty magnitude.

**Note:** The old `boundary` key is deprecated but still works (maps to `upper_boundary`).

#### 2. Hard Clipping Bounds (Optimizer Projection)

After each optimizer update, parameters are projected into a valid range
using `optax.projections.projection_box`.  This is a true hard wall — the
parameter value can never leave `[clip_min, clip_max]`.

Sigma and epsilon have **separate** configurable bounds:

```toml
[nn]
# Hard clipping for sigma (nm) — typical atomic radii are 0.1-0.5 nm
clip_sigma_min = 0.05     # default: 0.05 nm (0.5 Å)
clip_sigma_max = 2.0      # default: 2.0 nm (20 Å)

# Hard clipping for epsilon (kJ/mol) — typical LJ well depths are 0.1-10 kJ/mol
clip_epsilon_min = 0.001  # default: 0.001 kJ/mol
clip_epsilon_max = 100.0  # default: 100 kJ/mol
```

The projection is applied per-type: the first `n_sigma_train` entries of
`LJ_param` are clipped with `[clip_sigma_min, clip_sigma_max]`, and the
remaining `n_epsilon_train` entries with
`[clip_epsilon_min, clip_epsilon_max]`.

#### 3. Soft + Hard Interplay

Use **both** together for best results:

| Mechanism | Role | Gradient effect |
|-----------|------|-----------------|
| Soft boundary | Smooth repulsion near edges | Adds a penalty gradient → optimizer *learns* to avoid the wall |
| Hard clipping | Absolute safety net | No gradient signal (projection happens after update) |

Soft boundaries alone can be overridden by a strong loss gradient;
hard clipping alone gives no gradient signal (the optimizer doesn't
"know" it hit a wall and may keep pushing).  Together, the soft penalty
slows the approach and the hard clip prevents overflow.

**Recommended workflow:**
1. Set hard clips to the absolute physical limits (never want to cross).
2. Set soft boundaries *inside* the hard limits to start pushing back
   early.
3. Tune `boundary_S` and `boundary_C` so that the penalty is noticeable
   but doesn't dominate the physics-based loss.

#### 4. Practical Examples

**Standard protein-in-water SPC/E  (conservative defaults):**
```toml
[nn]
clip_sigma_min   = 0.05    # 0.5 Å — hydrogen-sized minimum
clip_sigma_max   = 2.0     # 20 Å — no atom that large
clip_epsilon_min = 0.001   # near-zero but non-zero (numerical safety)
clip_epsilon_max = 100.0   # generous ceiling

[nn.loss]
lower_boundary = 0.05
upper_boundary = 50.0
boundary_S     = 2
boundary_C     = 500
```

**Tight bounds for coordination metals  (e.g. Zn²⁺ crystal-field):**
```toml
[nn]
clip_sigma_min   = 0.15    # zinc coordination radius ≥ 0.15 nm
clip_sigma_max   = 0.35    # upper limit for ZN–S/N bond radii
clip_epsilon_min = 0.1     # meaningful well depth for Zn²⁺
clip_epsilon_max = 50.0

[nn.loss]
lower_boundary = 0.2       # soft push starts at 0.2
upper_boundary = 40.0
boundary_S     = 4          # steeper sigmoid for tighter control
boundary_C     = 1000
```

**Epsilon-only refinement  (sigma frozen):**
```toml
# When n_sigma_train = 0, only epsilon is trainable.
# Sigma clips are irrelevant; only epsilon bounds matter.
[nn]
clip_epsilon_min = 0.01
clip_epsilon_max = 20.0
```

**Choosing bounds:**
- **Sigma:** Atomic radii typically range from 0.1–0.5 nm. Set `clip_sigma_min`
  to prevent collapse (e.g., 0.05 nm) and `clip_sigma_max` to prevent blowup.
- **Epsilon:** LJ well depths are typically 0.1–10 kJ/mol. Very small values
  (< 0.001) cause numerical issues in finite-difference gradients (the FD
  epsilon floor is `max(|p|, 1e-3)`, so `|ε| = 0.003` gives FD step ≈ 3 × 10⁻⁷
  which is near f64 noise); very large values are unphysical.

---

## Distance Distribution Loss Template

Below is a template for a new loss function that matches **pairwise distance distributions**
between specific atom types (e.g., SZ–ZN and NZ–ZN coordination distances).
This would be added to `src/diff_md/losses.py`.

```python
def distance_distribution(
    # fmt: off
    model, system, key, start_temperature, comm,
    # System-specific args (from [nn.system_args.<name>])
    pair_selections,        # list of (type_name_A, type_name_B) pairs
    target_dist,            # (n_pairs, n_bins) target distributions
    data_range,             # (n_bins,) bin centers in nm
    # Loss args (from [nn.loss])
    metric,
    dist_weight=1.0,
    width_ratio=1.0,
    k_constraint=0.01,
    boundary=None, boundary_S=2, boundary_C=500,
    constraint=None,
):
    """Loss based on pairwise distance distributions (KDE) for specific atom-type pairs.

    Use case: optimizing LJ parameters to reproduce coordination distances,
    e.g. SZ-ZN (cysteine sulfur to zinc) and NZ-ZN (histidine nitrogen to zinc).
    """
    sgm_table, epsl_table, param_constraints, types = get_LJ_param(
        model, system.config, jnp.array(system.types)
    )

    trj, key, config = simulator(
        # fmt: off
        model, system.positions, system.velocities, types,
        system.masses, system.charges, sgm_table, epsl_table,
        key, system.topol, system.config, start_temperature,
    )

    comm_size = comm.Get_size()
    n_frames = len(trj["positions"])
    n_bins = data_range.size
    bin_size = float(data_range[1] - data_range[0])
    bandwidth = float(width_ratio * bin_size)
    n_pairs = len(pair_selections)

    # Skip initial equilibration frames
    n_skip = 0
    n_frames_adj = n_frames - n_skip

    kde_dists = jnp.zeros((n_pairs, n_bins))

    # Build index masks for each requested atom-type pair
    # pair_selections is a list of (idx_type_A, idx_type_B) after parsing
    # (nn_options converts type names to integer indices via name_to_type)

    for pos, box in zip(trj["positions"][n_skip:], trj["box"][n_skip:]):
        for p_idx, (type_a, type_b) in enumerate(pair_selections):
            # Select atoms of each type
            mask_a = system.types == type_a
            mask_b = system.types == type_b
            idx_a = jnp.where(mask_a, size=jnp.sum(mask_a))[0]
            idx_b = jnp.where(mask_b, size=jnp.sum(mask_b))[0]

            pos_a = pos[idx_a]  # (n_a, 3)
            pos_b = pos[idx_b]  # (n_b, 3)

            # Compute minimum-image pairwise distances
            # dx shape: (n_a, n_b, 3)
            dx = pos_a[:, None, :] - pos_b[None, :, :]
            dx = dx - box * jnp.round(dx / box)
            distances = jnp.sqrt(jnp.sum(dx ** 2, axis=-1))  # (n_a, n_b)

            # Flatten and apply KDE
            flat_dists = distances.ravel()
            gaussians = gaussian_kde(flat_dists, bw_method=bandwidth)
            kde_value = gaussians(data_range)
            kde_dists = kde_dists.at[p_idx].add(kde_value)

    # Average over frames and MPI ranks
    kde_dists = mpi4jax.allreduce(kde_dists, op=MPI.SUM, comm=comm)
    kde_dists /= comm_size * n_frames_adj

    # Normalize each distribution to unit area
    kde_dists = kde_dists / (jnp.sum(kde_dists, axis=1, keepdims=True) * bin_size + 1e-12)

    # Compute per-pair error and average
    error = jnp.sum(dist_weight * metric(kde_dists, target_dist, axis=1)) / n_pairs

    # Parameter constraints
    if constraint:
        error += constraint(model.LJ_param, k_constraint, param_constraints)

    # Boundary penalty
    if boundary:
        error += boundary_constraint(epsl_table, boundary_C, boundary_S, boundary)

    return error, (
        {"distance_distributions": kde_dists},
        trj,
        key,
        config,
        types,
    )
```

### Corresponding `training.toml` for the Distance Distribution Loss

```toml
[nn]
systems = ["zn_system"]
n_epochs = 50
equilibration = 0
teacher_forcing = false

[nn.optimizer]
name = "adam"
learning_rate = 0.001

[nn.loss]
name = "distance_distribution"
metric = "mse"
dist_weight = 1.0
width_ratio = 1.0
k_constraint = 0.01
constraint = "harmonic"
boundary = 25.0

[nn.system_args."zn_system"]
# Pairs whose distance distribution we want to match
# (converted to type indices internally via name_to_type)
pair_selections = [["SZ", "ZN"], ["NZ", "ZN"]]
# Reference distributions: .npy file with shape (n_pairs, n_bins)
target_dist = "reference_dist.npy"
# Bin centers in nm: .npy file with shape (n_bins,)
data_range = "dist_bins.npy"

# --- Option A: optimize SZ-ZN and NZ-ZN pairs separately (pair mode) ---
[nn.model]
LJ_param = [
    ["SZ",  "ZN",  2.50e-01, 0.500, "on_eps"],
    ["NZ",  "ZN",  2.60e-01, 0.400, "on_eps"],
    ["SZ",  "SZ",  3.10e-01, 0.250],
    ["SZ",  "NZ",  3.00e-01, 0.300],
    ["NZ",  "NZ",  3.20e-01, 0.350],
    ["ZN",  "ZN",  2.00e-01, 0.100],
]

# --- Option B: optimize sigma+epsilon of ZN, SZ, NZ as types (type mode) ---
# [nn.model]
# LJ_type_param = [
#     ["ZN",  2.00e-01, 0.100, "on_eps", "on_sigma"],
#     ["SZ",  3.10e-01, 0.250, "on_eps", "on_sigma"],
#     ["NZ",  3.20e-01, 0.350, "on_eps", "on_sigma"],
#     ["CT",  3.40e-01, 0.457],
#     ["N",   3.25e-01, 0.711],
#     ["O",   2.96e-01, 0.879],
# ]
```

### Integration Steps

To actually use the `distance_distribution` loss:

1. Add the function to `src/diff_md/losses.py` (copy the template above).
2. Add parsing logic in `nn_options.py` → `get_system_options()` to load
   `pair_selections` (convert names to type indices) and `target_dist` / `data_range`
   from `.npy` or `.xvg` files — similar to how `target_density` is loaded for
   `density_and_apl`.
3. Prepare reference data:
   - `reference_dist.npy`: shape `(n_pairs, n_bins)` — e.g., from a GROMACS RDF
     or from `gmx rdf` output converted to `.npy`.
   - `dist_bins.npy`: shape `(n_bins,)` — bin centers in nm.
4. Run: `diff_md optimize -f input.h5 -p topol.toml -c options.toml -m training.toml`

---

## GPU Acceleration for `diff_md optimize`

JAX automatically uses a GPU if one is available and `jaxlib[cuda]` is installed.
No code changes are required to move from CPU to GPU — all `jnp` operations, `lax.scan`,
`vmap`, and FFTs dispatch to the GPU transparently.

However, several patterns in the current codebase limit GPU throughput. The table below
ranks them by impact.

### Bottleneck Summary

| Priority | Issue | Where | Impact |
|----------|-------|-------|--------|
| **P0** | Python for-loop over trajectory frames in loss functions | `losses.py` | Each frame is a separate kernel launch; no fusion |
| **P0** | Trajectory accumulated via Python list `.append()` | `simulate.py` | Host–device transfer after each `lax.scan` chunk |
| **P0** | No gradient checkpointing on PME reciprocal pass | `nonbonded.py` | Memory blowup for large systems during backprop |
| **P1** | `step()` not JIT-compiled as a whole | `optimize.py` | `value_and_grad` + optimizer update + projection all separate |
| **P2** | Initial neighbor list built with NumPy on CPU | `neighbor_list.py` | One-time cost at simulation start |

### Detailed Recommendations

#### 1. Vectorize loss frame loops (`losses.py`) — HIGH IMPACT

Current pattern (e.g., `density_and_apl`):
```python
for pos, box in zip(trj["positions"], trj["box"]):
    kde_density = lateral_density_kde(kde_density, ...)
```

This launches a separate GPU kernel per frame. Replace with `vmap` + `jnp.sum`:
```python
def _frame_density(pos, box):
    com = _compute_com(pos[fixed_sel, 2], box[2])
    centered = _center(pos[:, 2], com, box[2])
    return lateral_density_kde(jnp.zeros(...), centered, ...)

kde_all = jax.vmap(_frame_density)(
    jnp.stack(trj["positions"]), jnp.stack(trj["box"])
)
kde_density = jnp.sum(kde_all, axis=0)
```

This fuses all frames into one batched kernel — typically **5–20× faster on GPU**.
The same applies to `radius_of_gyration` and all its variants.

#### 2. Pre-allocate trajectory arrays (`simulate.py`) — HIGH IMPACT

Currently, `trj["positions"].append(pos)` inside the scan loop triggers a
device-to-host transfer at each print step. Instead:

```python
# Pre-allocate on GPU
n_frames = n_steps // n_print + 1
trj_pos = jnp.zeros((n_frames, n_atoms, 3))
trj_box = jnp.zeros((n_frames, 3))

# Inside lax.scan, write frames at computed indices
trj_pos = trj_pos.at[frame_idx].set(positions)
```

This keeps all trajectory data on GPU until the scan completes, avoiding
O(n_frames) host–device round trips.

#### 3. Gradient checkpointing on PME (`nonbonded.py`) — ✅ IMPLEMENTED

The reciprocal-space energy computation stores the full mesh + FFT intermediates
for backprop. For >10k atoms this can exhaust GPU memory.  Now wrapped with
`@jax.checkpoint` on `_recip_energy` and `_dip_recip_energy` (both in
`nonbonded.py`), plus `jax.checkpoint(_md_step)` on the `lax.scan` body in
`simulate.py`.  The diagnostic potential-field forward pass is also skipped
during training (`compute_potential=False`) since no loss function uses it.

This trades ~2× recomputation for O(1) memory on the backward pass.

**Speed vs memory trade-off:**  The `jax.checkpoint` on `_md_step` recomputes the
entire step body during backprop (~2× compute cost).  The nested `@checkpoint` on
`_recip_energy` inside the PME function adds one additional PME forward pass per
step.  To maximise speed at the expense of more GPU memory, remove the outer
checkpoint:

```python
# In simulate.py, replace:
_md_step_ckpt = jax.checkpoint(_md_step)
# With:
_md_step_ckpt = _md_step
```

#### 4. JIT the full optimization step (`optimize.py`)

The `step()` closure calls `value_and_grad`, optax update, and projection
in separate Python calls. Wrapping the whole thing in `@jax.jit`:

```python
@jax.jit
def jit_step(params, opt_state, key):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ...)
    updates, new_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params = optax.projections.projection_box(new_params, lower, upper)
    return new_params, new_state, loss, aux
```

This compiles the entire forward+backward+update into a single XLA graph,
eliminating host round trips between sub-operations. First call is slow
(compilation), subsequent calls are fast.

#### 5. Multi-GPU via `jax.pmap` (future)

The current MPI-based parallelism (`mpi4jax`) works across nodes but doesn't
exploit multiple GPUs on the same node natively. For single-node multi-GPU:

```python
# Replace mpi4jax.allreduce with:
grads = jax.lax.pmean(grads, axis_name="devices")
# And wrap step() with pmap:
pmap_step = jax.pmap(jit_step, axis_name="devices")
```

This is a larger refactor but enables efficient multi-GPU gradient averaging
without MPI overhead on a single node.


#COMANDO x LANCIARE su olivia

srun -n 4 --gpus-per-task=1


#INFO Nuova Loss x ZINCO

What it does, step by step
Goal: At each epoch, run an MD trajectory, measure how far each SZ/NZ/SD atom is from each Zn atom, build a distance distribution for each pair type, and compare it to your QM/MM reference.

1. Run the simulation
Same as every other loss — build LJ tables from the current trainable parameters, run the simulator, get back a trajectory (trj) with positions+boxes for every saved frame.

2. Collect distances
For every frame of the trajectory and for each pair type (SZ-Zn, NZ-Zn, SD-Zn):

The helper _pairwise_distances_pbc takes all SZ atoms and all Zn atoms, computes all Na × Nb minimum-image distances (properly wrapped for periodic boxes), and returns them as a flat array.
These distances are appended to a list.
After looping over all frames, all distances for pair p are concatenated into one big 1-D array. For example, if you have 2 SZ atoms, 1 Zn atom, and 50 frames → 2×1×50 = 100 distance values for the SZ-Zn pair.

3. Build a KDE distribution
For each pair type, a Gaussian KDE (kernel density estimate) is built from all those concatenated distances. This gives a smooth probability distribution curve — essentially a histogram smoothed by Gaussians — evaluated on the same bin centres as your reference XVG (data_range).

This is the key: one KDE per pair type, built from the entire trajectory of this epoch (not per-frame).

4. Average across MPI ranks
Each MPI rank ran its own independent trajectory (different PRNG seed). The KDEs are summed via allreduce and divided by the number of ranks → you get a rank-averaged distribution.

5. Compute the error
For each pair type, compare the simulated KDE to the corresponding column from your QM/MM reference XVG using the chosen metric (MSE). Sum the per-pair errors and divide by the number of pairs (3 in your case) so the loss scale doesn't depend on how many pairs you define.

6. Gradients flow back
Because everything is JAX — the LJ parameters → forces → positions → distances → KDE → error is a differentiable chain. value_and_grad gives you the gradient of this error with respect to your trainable LJ σ and ε. The optimizer updates them, and next epoch the MD runs with slightly different LJ parameters, hopefully producing distributions closer to the QM/MM reference.

What was fixed
The initial version had a subtle but important bug: it built a KDE per frame (from just a handful of distances, e.g. 2 for a single Zn) and averaged those KDEs. With so few data points per frame the KDE is essentially noise — a few spikes rather than a smooth distribution. By contrast, the fix collects all distances across all frames first, then builds one well-sampled KDE from the full trajectory. This is physically correct: you want the trajectory-averaged distance distribution, not the average of per-frame distributions.

The simpler coordination_distance loss (mean-distance variant) had a normalisation order bug: it divided by n_frames before the MPI allreduce, which gives a wrong result. Now it does a single / (comm_size * n_frames) after allreduce — consistent with every other loss in the codebase.

In summary
Each epoch:
Your LJ params → MD trajectory → for each of {SZ-Zn, NZ-Zn, SD-Zn}: collect all pairwise distances → build one smooth distribution (KDE) → compare to QM/MM reference → gradient → update LJ params.



TEMPLATE:
[nn]
# Hard clipping bounds only for range definition
clip_sigma_min = 0.05
clip_sigma_max = 2.0
clip_epsilon_min = 0.001
clip_epsilon_max = 100.0

[nn.loss]
# Soft boundaries (loss penalties) to dug the Loss f. optimization away from these "bad" values
upper_boundary = 50.0
lower_boundary = 0.05
boundary_S = 2
boundary_C = 500
constraint = "harmonic"
k_constraint = 0.01

# Added cache cleaning!!! sotto grad_method, ogni quante epoche fare pulito
clear_xla_cache = 0        (prend interi 1 5 10 ogni quante epoche!!!)



TOML new option options.toml
nlist_capacity_multiplier = 1.25   # default now; increase to 1.5 if overflow persists

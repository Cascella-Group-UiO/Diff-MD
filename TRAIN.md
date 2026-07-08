# Diff-HyMD Training Guide

Diff-HyMD (`diff_md optimize`) turns molecular-dynamics simulation into a
*differentiable* function of the force-field parameters. It runs short MD
trajectories inside an automatic-differentiation graph (JAX), computes a
physical observable from each trajectory (e.g. a density profile, a
radius-of-gyration distribution, a metal–ligand coordination distance),
compares it to a reference target, and backpropagates that error **through the
entire simulation** to update the parameters. The result is a force field whose
simulated observables match your reference data.

Today the trainable parameters are the **Lennard-Jones** σ and ε terms;
everything else in the force field (bonds, angles, dihedrals, electrostatics) is
held fixed and enters the gradient only through the dynamics. Optimization is
driven by a loss function chosen in `training.toml`, with optional soft
penalties and hard bounds that keep parameters physical.

This guide covers how to declare which LJ parameters to train, the available
loss functions and metrics, how to constrain parameters, and the
coordination-distance losses used for metal sites. For the simulation engine and
runtime options see `README.md`; for the optimization internals (gradient
methods, checkpointing, restart) see `OPTIMIZE.md`.

**Typical command:**

```bash
diff_md optimize -f input.h5 -p topol.toml -c options.toml -m training.toml
```

- `-f input.h5` — initial coordinates, velocities, and box
- `-p topol.toml` — topology (bonded terms, exclusions, atom types)
- `-c options.toml` — runtime/simulation options (cutoffs, thermostat, neighbor list)
- `-m training.toml` — **the training configuration** (the subject of this guide)

## Table of Contents

- [How LJ Parameters Are Specified in `training.toml`](#how-lj-parameters-are-specified-in-trainingtoml)
  - [Mode 1: Pair Mode (`LJ_param`)](#mode-1-pair-mode-lj_param)
  - [Mode 2: Type Mode (`LJ_type_param`)](#mode-2-type-mode-lj_type_param)
  - [How the Internal Mapping Works](#how-the-internal-mapping-works)
- [Concrete Examples for Zn Coordination (SZ, NZ, ZN)](#concrete-examples-for-zn-coordination-sz-nz-zn)
- [Loss Function Reference](#loss-function-reference)
  - [Available Losses](#available-losses)
  - [Available Metrics](#available-metrics)
  - [Constraints and Boundaries](#constraints-and-boundaries)
- [Coordination Losses for Metal Sites](#coordination-losses-for-metal-sites)
  - [Distance distribution loss](#distance-distribution-loss)
  - [Mean distance loss](#mean-distance-loss)
  - [Tetrahedral order losses](#tetrahedral-order-losses)
  - [Writing your own loss](#writing-your-own-loss)
- [Useful Training Options](#useful-training-options)

---

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

| Name | Target Observable | Key System Args |
|------|-------------------|-----------------|
| `density_and_apl` | Lateral density profile + area per lipid | `z_range`, `com_type`, `n_lipids`, `target_density`, `target_apl` |
| `radius_of_gyration` | Mean Rg | `n_chains`, `chain_indices`, `chain_masses`, `target_rg` |
| `radius_of_gyration_dist` | Rg probability distribution (KDE) | `n_chains`, `chain_indices`, `chain_masses`, `data_range`, `target_dist` |
| `radius_of_gyration_median` | Median Rg across replicas | same as `radius_of_gyration` |
| `radius_of_gyration_filter_repls` | Filtered mean Rg | same as `radius_of_gyration` |
| `radius_of_gyration_and_end_to_end` | Rg + end-to-end distance | same + `target_end_to_end` |
| `coordination_distance` | Mean metal–ligand distance(s) | `coord_pairs`, `target_distances` |
| `coordination_distance_dist` | Metal–ligand distance distribution (KDE) | `coord_pairs`, `data_range`, `target_dist` |
| `coordination_tetrahedral` | Tetrahedral order *q* + site distances | per-site metal/ligand defs, `target_q`, `target_site_distances` |
| `coordination_tetrahedral_dist` | Tetrahedral *q* + distance distributions | per-site defs, `data_range`, `target_dist`, `target_q` |

The Rg observables are mass-weighted about the center of mass (matching
`gmx gyrate`). The coordination losses are documented in detail in
[Coordination Losses for Metal Sites](#coordination-losses-for-metal-sites).

### Available Metrics

Specified in `[nn.loss] metric = "..."`:

| Metric | Description |
|--------|-------------|
| `mse`  | Mean squared error |
| `rmse` | Root mean squared error |
| `smape`| Symmetric mean absolute percentage error |
| `l2e`  | L2 norm of error |
| `wasserstein_1d` | 1-D Wasserstein (earth-mover) distance — well suited to comparing distributions |

The metric name maps directly to the function of the same name in
`src/diff_md/losses.py`.

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

## Coordination Losses for Metal Sites

These losses optimize LJ parameters so that the **coordination geometry** of a
metal centre (and its ligands) matches a reference, typically from QM/MM. They
cover mean distances, full distance distributions, and the tetrahedral order
parameter, and all of them accept the same constraint/boundary options described
above.

### Distance distribution loss

`coordination_distance_dist` matches the **distribution** of distances between
two atom types (e.g. metal–ligand) against a reference. For each pair type it
builds one Gaussian KDE from all distances collected over the whole trajectory
(and all MPI replicas), then compares it to the target with the chosen metric.

```toml
[nn]
systems = ["zn_system"]
n_epochs = 50

[nn.optimizer]
name = "adam"
learning_rate = 0.001

[nn.loss]
name = "coordination_distance_dist"
metric = "mse"            # or "wasserstein_1d" for distribution matching
dist_weight = 1.0
width_ratio = 1.0         # KDE bandwidth = width_ratio * bin_size
k_constraint = 0.01
constraint = "harmonic"
upper_boundary = 50.0
lower_boundary = 0.05

[nn.system_args."zn_system"]
coord_pairs = [["SZ", "ZN"], ["NZ", "ZN"]]   # type-name pairs; converted to indices internally
target_dist = "reference_dist.xvg"            # .xvg/.npy: row 0 = bin centres (nm), rows 1.. = per-pair target densities

[nn.model]
LJ_param = [
    ["SZ", "ZN", 2.50e-01, 0.500, "on_eps"],
    ["NZ", "ZN", 2.60e-01, 0.400, "on_eps"],
    ["SZ", "SZ", 3.10e-01, 0.250],
    ["NZ", "NZ", 3.20e-01, 0.350],
    ["ZN", "ZN", 2.00e-01, 0.100],
]
```

**What it computes, per epoch:**

1. Build LJ tables from the current trainable parameters and run the simulator,
   producing a trajectory of positions and boxes.
2. For each pair type, compute all minimum-image distances between the two type
   groups across **all** frames (properly wrapped for periodic boxes) and
   concatenate them.
3. Build one Gaussian KDE per pair type from the full set of distances
   (trajectory-averaged, *not* per-frame) and evaluate it on the reference bin
   centres `data_range`.
4. Sum the KDEs across MPI ranks and normalize each to unit area.
5. Average the per-pair error against the reference.
6. Because the whole chain — LJ parameters -> forces -> positions -> distances
   -> KDE -> error — is differentiable, `value_and_grad` updates σ/ε toward the
   reference each epoch.

> **Why the trajectory-wide KDE matters:** building a KDE from the handful of
> distances in a *single* frame (e.g. one Zn atom) is essentially noise.
> Collecting all distances first yields a well-sampled, physically meaningful
> distribution — the trajectory-averaged distance distribution, not the average
> of per-frame distributions.

### Mean distance loss

`coordination_distance` is the cheaper mean-distance variant: instead of a full
distribution it matches the **mean** metal–ligand distance(s) to
`target_distances`. Useful when you only have reference mean coordination
distances. Optional `data_range` / `target_dist` can be supplied purely to emit
a diagnostic KDE in the output.

```toml
[nn.loss]
name = "coordination_distance"
metric = "mse"

[nn.system_args."zn_system"]
coord_pairs = [["SZ", "ZN"], ["NZ", "ZN"]]
target_distances = [0.23, 0.21]    # nm, one per pair
```

### Tetrahedral order losses

`coordination_tetrahedral` and `coordination_tetrahedral_dist` extend
coordination matching to the **tetrahedral order parameter *q*** of a metal site
together with its ligand distances, defined per site (a metal plus its ligand
groups). Use these when the *geometry* of the coordination shell — not just the
distances — must match the reference. The per-site topology (metal index, ligand
indices, group labels) is given in `[nn.system_args]`; see
`src/diff_md/losses.py` for the exact per-site arguments.

### Writing your own loss

All losses live in `src/diff_md/losses.py` and share one signature:

```python
def my_loss(model, system, key, start_temperature, comm,  # always provided by optimize.py
            arg_from_system_args,                          # from [nn.system_args.<name>]
            arg_from_loss):                                # from [nn.loss]
    ...
    return error, (output_dict, trj, key, config, types)
```

To add one: implement the function, add its argument parsing in `nn_options.py`
(convert type names to indices, load `.xvg` / `.npy` references), then select it
with `[nn.loss] name = "my_loss"`.

---

## Useful Training Options

A few `training.toml` knobs that matter in practice:

| Option | Section | Effect |
|--------|---------|--------|
| `n_epochs` | `[nn]` | Number of optimization epochs |
| `equilibration` | `[nn]` | Equilibration steps run before each scored trajectory |
| `teacher_forcing` | `[nn]` | Reuse the previous epoch's final state as the next start (vs. re-equilibrating) |
| `train_sigma` | `[nn]` | In type mode with no explicit flags, also train σ (default `false`) |
| `grad_method` | `[nn.loss]` | Gradient method: `"reverse"` (reverse-mode AD, default), `"jvp"` (forward-mode, lower memory for long runs), or `"finite_diff"` |
| `clear_xla_cache` | `[nn.loss]` | Clear the XLA compilation cache every N epochs (`0` = never); bounds memory on long runs |
| `clip_sigma_min` / `clip_sigma_max` | `[nn]` | Hard bounds on σ (nm) — see [Constraints and Boundaries](#constraints-and-boundaries) |
| `clip_epsilon_min` / `clip_epsilon_max` | `[nn]` | Hard bounds on ε (kJ/mol) |

The neighbor-list capacity is a runtime option set in `options.toml`:

```toml
nlist_capacity_multiplier = 1.25   # raise to 1.5 if you see neighbor-list overflow warnings
```


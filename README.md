# Diff-HyMD
∂-HyMD (read diff-HyMD) is built on top of [∂-HyMD_hHPF](https://github.com/Cascella-Group-UiO/).
The main goal is to automatically learn force field parameters while running differentiable molecular dynamics simulations.
Instead of using the hybrid particle-field Hamiltonian, this version uses regular force field functions.

#To read more about ∂-HyMD check the paper [here](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00564).

For force-field **training** (which LJ parameters to optimize, loss functions,
constraints) see [`TRAIN.md`](TRAIN.md); for the optimization internals
(gradient methods, checkpointing, restart) see [`OPTIMIZE.md`](OPTIMIZE.md).

## Table of Contents

- [Installation](#installation)
- [Example usage](#example-usage)
- [Optimization](#optimization)
- [Parameter Constraints and Bounds](#parameter-constraints-and-bounds)
- [Units for TOML inputs](#units-for-toml-inputs)
- [PME/Ewald Accuracy (`sigma` and `rc`)](#pmeewald-accuracy-sigma-and-rc)
- [B-spline PME Interpolation](#b-spline-pme-interpolation)
- [LJ Force-Shift and NVE Ensemble](#lj-force-shift-and-nve-ensemble)
- [Integrator](#integrator)
- [NPT Barostat (Constant Pressure)](#npt-barostat-constant-pressure)
- [Neighbor List And Verlet Skin](#neighbor-list-and-verlet-skin)
- [Neighbor List Rebuild Logic (detailed)](#neighbor-list-rebuild-logic-detailed)
- [jax-md Neighbor List (alternative backend)](#jax-md-neighbor-list-alternative-backend)
- [Testing](#testing)
- [Whole-molecule unwrapping for visualisation output](#whole-molecule-unwrapping-for-visualisation-output)
- [Optimization call graph](#optimization-call-graph)
- [Available loss functions](#available-loss-functions)
- [Recent Changes](#recent-changes)

## Installation
> **Note**:
> If installing on Saga or Betzy you need to first load the `python` and `openmpi` modules
> ```terminal
> module load Python/3.11.3-GCCcore-12.3.0
> module load OpenMPI/4.1.5-GCC-12.3.0
> ```
> and then proceed with the installation.

Clone the repo on your machine and create a virtual enviroment inside a directory `<dir>` of your choice
```terminal
cd Diff-MD
python -m venv --upgrade-deps <dir>
```
Then you can simply install the package with
```terminal
source <dir>/bin/activate
pip install .
```
## Example usage
To run a simple MD simulation you can use
```terminal
cd examples
diff_md mdrun -f dppc/input.h5 -p dppc/topol.toml -c dppc/options.toml -o dppc/simulation -v
```
To continue (append to) an existing trajectory:
```terminal
diff_md mdrun -f dppc/simulation.h5 -p dppc/topol.toml -c dppc/options.toml -o dppc/simulation --append -v
```

### Restarting a simulation (`--append`)

The `--append` flag performs an **exact restart**: positions and velocities
are read from the last frame of the existing output trajectory (not from the
original input file), and the step counter and simulation time continue from
where they left off.

This means that, after the restart, there is no energy jump at the boundary
between the old and the new frames — the simulation is thermodynamically
continuous.

**Auto-rollback:** If the output trajectory contains trailing corrupt frames
(all-zero positions from a pre-allocated but never-written H5 chunk, or NaN
values from an interrupted write), Diff-MD automatically scans backwards and
restarts from the last valid frame.  A warning is logged with the number of
skipped frames.  If velocities at the chosen frame also contain NaN, they
are re-initialised from a Maxwell-Boltzmann distribution.

The same NaN/zero validation is applied when reading input coordinate files
(`System.constructor`) and inside the `reconnect_for_append` H5MD handler.

Requirements:
- The output file (`-o`) must point to an existing `.h5` trajectory that was
  produced by a previous run.
- Topology (`-p`) and options (`-c`) must match the original run.
- You can change `n_steps` to extend the total number of simulation steps.

Example — run 100 000 steps, then extend by another 100 000:
```terminal
# First run
diff_md mdrun -f input.h5 -p topol.toml -c options.toml -o sim -v

# Continuation (reads positions/velocities from sim.h5)
diff_md mdrun -f input.h5 -p topol.toml -c options.toml -o sim --append -v
```

### Last-frame snapshot (`finish.h5`)

At the end of every `mdrun` simulation, Diff-MD automatically writes a
standalone **`finish.h5`** file in the output directory.  This file contains
only the final frame (positions, velocities, box, types, names, masses, and
charges) in H5MD format and can be used directly as the input coordinate
file for a new simulation:

```terminal
# Original run — produces sim.h5 and finish.h5
diff_md mdrun -f input.h5 -p topol.toml -c options.toml -o sim -v

# New run starting from the last frame of the previous simulation
diff_md mdrun -f sim/finish.h5 -p topol.toml -c options.toml -o sim2 -v
```

This is particularly useful when you want to:
- Continue a simulation with **different options** (temperature, time step, etc.)
- Branch several independent runs from the same equilibrated state
- Archive a lightweight restart point without keeping the full trajectory

> **Note:** `finish.h5` is always overwritten on each run.  If you used
> `--append`, the snapshot corresponds to the last frame of the extended
> trajectory.  For NPT simulations, the box dimensions are taken from the
> barostat state at the final step.

Instead, to optimize force field parameters you can run
```terminal
cd examples
sed -i -e s/10000/200/ dppc/options.toml # Use smaller number of steps when training
destdir=`pwd`
diff_md optimize -f input.h5 -p topol.toml -c options.toml -o dppc/train -m dppc/training.toml -d $destdir -v
```
## Optimization
Optimization requires a bit more work, so carefully check `dppc/training.toml` for all the available options.

Inside `training.toml` we need to specify a `system` list.
The elements of this list are directories that each contain the inputs to `diff_md`, with the same name provided in the command line
(in the example above, these are `input.h5`, `topol.toml`, and `options.toml`).
The system directories paths are relative to the working directory path from which you call `diff_md`.

It is also possible to run multiple replicas of the optimzation in parallel, by using `mpirun`
```terminal
mpirun -n 4 diff_md optimize ...
```

### MPI Parallel Optimization

Diff-MD uses a **data-parallel** strategy: each MPI rank runs an independent
MD trajectory on its own GPU (unique PRNG seed) and the results are reduced
so the optimizer sees a single, averaged gradient per epoch.

#### How the reduction works

1. **Forward pass (inside the loss function):**
   Each rank runs `simulator()` independently, accumulates observables
   (density, area-per-lipid, Rg, ...) from its own trajectory, and then
   calls `mpi4jax.allreduce(obs, MPI.SUM)`.  The result is divided by
   `comm_size`, so every rank holds the **same** averaged observable
   and computes the **same** scalar loss.

2. **Backward pass (automatic through `jax.grad`):**
   The backward of `allreduce(SUM)` is the **identity** (no MPI call).
   Each rank's gradient therefore already carries a `1/N` factor from
   the forward-pass division by `comm_size`.

3. **Gradient allreduce (in `optimize.py`):**
   `grads = jax.tree.map(lambda g: mpi4jax.allreduce(g, MPI.SUM), grads)`
   sums the per-rank gradient contributions.  Because each term is
   already scaled by `1/N`, the sum equals the **true gradient** —
   invariant of the number of ranks.

This means **no learning-rate rescaling is needed** when you change the
number of ranks (4 → 8 → 16 ...).  The gradient magnitude stays the same.

#### Single-node example (4 GPUs)

```terminal
srun -n 4 --ntasks-per-node=4 --gpus-per-task=1 \
     diff_md optimize -f input.h5 -p topol.toml -c options.toml \
     -m training.toml -o train -d $destdir -v
```

#### Multi-node scaling (e.g. 3 nodes × 4 GPUs = 12 parallel replicas)

Just increase `-n` and `--nodes`; the code is fully node-count agnostic:
```terminal
srun --nodes=3 -n 12 --ntasks-per-node=4 --gpus-per-task=1 \
     diff_md optimize -f input.h5 -p topol.toml -c options.toml \
     -m training.toml -o train -d $destdir -v
```

#### Full SLURM batch script example (OLIVIA)

```bash
#!/bin/bash
#SBATCH --job-name=diff-md-train
#SBATCH --partition=boost_usr_prod   # adjust to your cluster
#SBATCH --nodes=2                    # number of nodes
#SBATCH --ntasks-per-node=4          # 4 MPI ranks per node (= 4 GPUs)
#SBATCH --gpus-per-task=1            # 1 GPU per rank
#SBATCH --cpus-per-task=8            # 1/4 of the node's CPU cores per rank
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out

module load python cuda openmpi      # adjust to your module system
source /path/to/venv/bin/activate

# Recommended environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # let JAX grow GPU memory on demand

srun diff_md optimize \
     -f input.h5 -p topol.toml -c options.toml \
     -m training.toml -o train -d $PWD -v
```

> **Note on GPU-aware MPI:** If your cluster provides a GPU-aware MPI
> library (e.g. `OpenMPI +cuda`), `mpi4jax` can transfer data directly
> between GPUs via NVLink/IB without staging through host memory.
> This is optional — the code works correctly either way.

#### GPU pinning

Diff-MD assigns one GPU to each MPI rank **before** JAX is imported.
The detection chain (in order of precedence) is:

1. `CUDA_VISIBLE_DEVICES` already set (e.g. by `--gpus-per-task`) → used as-is
2. `SLURM_LOCALID` environment variable
3. `OMPI_COMM_WORLD_LOCAL_RANK` (OpenMPI)
4. `MPI_LOCALRANKID` (MPICH / Intel MPI)
5. Portable MPI-3 `comm.Split_type(COMM_TYPE_SHARED)` fallback

All methods return the **node-local** rank (0 .. GPUs_per_node−1),
so the mapping is correct regardless of how many nodes are used.

Finally, the program automatically checkpoints the state of the gradients and the parameters after each epoch.
These checkpoints are saved in the `step_#/cpt` directories.
It's possible to restart the optimization from a given checkpoint by simply passing that directory to `diff_md`:
```terminal
diff_md optimize ... --restart step_600/cpt
```

When running with MPI, only rank 0 reads the checkpoint file and broadcasts
the restored epoch, parameters, and optimizer state to all other ranks.
This avoids filesystem-consistency issues on multi-node clusters where NFS
views may briefly differ.

#### Per-rank input conformations

By default every MPI rank reads the same coordinate file (e.g. `input.h5`)
and diverges only through its unique PRNG seed.  For better sampling,
you can provide **rank-specific** coordinate files so that each rank starts
from a different conformation.

Place files named `<stem>_<RANK>.h5` next to the shared file in each
system directory.  `<RANK>` is the zero-padded four-digit global MPI rank.
If a rank-specific file exists it is used automatically; otherwise the
shared file is loaded.  Topology (`-p`) and options (`-c`) remain shared
across all ranks.  Output trajectories are already rank-tagged
(`step_<epoch>/<rank:04d>/trajectory.h5`).

#### Full example: 8 ranks on 2 nodes with 8 different starting conformations

**Step 1 — directory layout**

Put all shared inputs plus the 8 per-rank coordinate files into one
system directory (here called `my_system`).  You can use symlinks if the
original files live elsewhere:

```
project/
├── training.toml
└── my_system/
    ├── topol.toml             # shared topology
    ├── options.toml           # shared MD options
    ├── input.h5               # optional fallback (any conformation)
    ├── input_0000.h5          # ← rank 0 starting conformation
    ├── input_0001.h5          # ← rank 1
    ├── input_0002.h5
    ├── input_0003.h5
    ├── input_0004.h5
    ├── input_0005.h5
    ├── input_0006.h5
    ├── input_0007.h5          # ← rank 7
    └── reference_aa.xvg       # target data (if needed by the loss)
```

Create the rank files from 8 existing conformations:

```bash
for i in $(seq 0 7); do
    cp /path/to/confs/conf_${i}.h5 my_system/$(printf 'input_%04d.h5' $i)
done
```

Or with symlinks:

```bash
for i in $(seq 0 7); do
    ln -s $(realpath /path/to/confs/conf_${i}.h5) my_system/$(printf 'input_%04d.h5' $i)
done
```

**Step 2 — ``training.toml``** (relevant excerpts)

```toml
[nn]
systems = ["my_system"]   # single system dir; per-rank files are automatic
n_epochs = 150
teacher_forcing = false
shuffle = false

[nn.optimizer]
name = "adabelief"
learning_rate = 0.05

[nn.loss]
name = "density_and_apl"
metric = "mse"

# Key MUST match the directory name in "systems" above
[nn.system_args.my_system]
com_type = "C1"
n_lipids = 64
target_apl = 0.633
target_density = "reference_aa.xvg"

[nn.model]
LJ_param = [
  # ["type_A", "type_B", sigma, epsilon, flags...],
]
```

**Step 3 — SLURM batch script** (2 nodes × 4 GPUs = 8 ranks)

```bash
#!/bin/bash
#SBATCH --job-name=diff-md-8rank
#SBATCH --partition=boost_usr_prod   # adjust to your cluster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # 4 MPI ranks per node (= 4 GPUs)
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out

module load python cuda openmpi      # adjust to your module system
source /path/to/venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /path/to/project                  # directory containing training.toml

srun diff_md optimize \
     -f input.h5 \
     -p topol.toml \
     -c options.toml \
     -m training.toml \
     -o train \
     -d $PWD/results \
     -v
```

`-f input.h5` is the shared fallback name.  Each rank automatically checks
for `input_<rank:04d>.h5` inside the system directory first.

**Output structure** after epoch 5:

```
results/
├── loss.dat
├── step_5/
│   ├── training.toml          # current parameters (written by rank 0)
│   ├── cpt/                   # Orbax checkpoint (written by rank 0)
│   ├── 0000/trajectory.h5     # rank 0 trajectory
│   ├── 0001/trajectory.h5     # rank 1
│   │   ...
│   └── 0007/trajectory.h5     # rank 7
└── final.toml                 # written after the last epoch
```

**Step 4 — restart from a checkpoint**

Checkpoints are saved at the end of every epoch to `results/step_<N>/cpt/`.
To restart from epoch 42:

```bash
srun diff_md optimize \
     -f input.h5 -p topol.toml -c options.toml \
     -m training.toml -o train -d $PWD/results \
     --restart results/step_42/cpt \
     -v
```

Restart behaviour:

| Step | What happens |
|---|---|
| Startup | All 8 ranks re-read their per-rank coordinate files |
| `--restart` path | Automatically converted to an absolute path (required by Orbax) |
| Checkpoint load | **Rank 0 only** reads the checkpoint via Orbax |
| Broadcast | Rank 0 `comm.bcast()`s `start_epoch`, `params`, `opt_state` to ranks 1–7 |
| Training loop | Resumes from epoch 43 with the restored parameters |

#### Recovering from a crash mid-epoch

If the job dies (OOM, node failure, walltime limit) during epoch N, the
checkpoint for epoch N has **not** been written yet — only `step_<N-1>/cpt/`
exists.  Recovery steps:

1. **Find the last valid checkpoint:**
   ```bash
   ls -d results/step_*/cpt | sort -t_ -k2 -n | tail -1
   # e.g. results/step_14/cpt
   ```

2. **Restart the same job** with `--restart` pointing to that directory.
   Use the exact same `srun` / `mpirun` invocation (same number of ranks):
   ```bash
   # 4-GPU single-node example
   srun -n 4 --ntasks-per-node=4 --gpus-per-task=1 \
        diff_md optimize \
        -f input.h5 -p topol.toml -c options.toml \
        -m training.toml -o train -d $PWD/results \
        --restart results/step_14/cpt \
        -v
   ```

3. **Changing the number of ranks** at restart is safe.  The checkpoint
   stores only the (shared) parameters and optimizer state, not per-rank
   trajectories.  Each rank re-reads its own coordinate file and gets a
   fresh PRNG seed derived from `--seed + rank`.

4. **Automated restart in a SLURM script** — resubmit the job automatically
   on timeout by picking the latest checkpoint:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=diff-md-train
   #SBATCH --partition=boost_usr_prod
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-task=1
   #SBATCH --cpus-per-task=8
   #SBATCH --time=24:00:00
   #SBATCH --output=train_%j.out
   #SBATCH --signal=B:USR1@120   # send signal 120 s before walltime

   module load python cuda openmpi
   source /path/to/venv/bin/activate

   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

   DESTDIR=$PWD/results

   # Find latest checkpoint (empty string if first run)
   LATEST_CPT=$(ls -d "$DESTDIR"/step_*/cpt 2>/dev/null | sort -t_ -k2 -n | tail -1)
   RESTART_FLAG=""
   if [[ -n "$LATEST_CPT" ]]; then
       echo "Restarting from $LATEST_CPT"
       RESTART_FLAG="--restart $LATEST_CPT"
   fi

   # Trap walltime signal → resubmit
   requeue() { scontrol requeue "$SLURM_JOB_ID"; }
   trap requeue USR1

   srun diff_md optimize \
        -f input.h5 -p topol.toml -c options.toml \
        -m training.toml -o train -d "$DESTDIR" \
        $RESTART_FLAG -v
   ```

   Submit once; the job will keep restarting itself until all epochs are done
   or the checkpoint indicates convergence.

> **Note on `teacher_forcing`:** With `teacher_forcing = false` (recommended
> for this workflow), each epoch starts a fresh trajectory from the original
> per-rank coordinates.  With `teacher_forcing = true`, each epoch continues
> from the previous epoch's final positions — after a restart, the first
> epoch will start from the *original* per-rank files (not from the last
> saved trajectory).  Use `equilibration` to warm up in that case.

## Parameter Constraints and Bounds

During optimization, trainable LJ parameters can drift to unphysical values
(negative epsilon, unrealistically large sigma, etc.).  Diff-MD provides two
independent mechanisms to prevent this: a **soft harmonic constraint** added
to the loss, and a **hard box projection** applied after each optimizer step.

### 1. Hard box projection (always active)

At the end of every optimizer step, parameters are projected back into a
physical interval using `optax.projections.projection_box`:

```
LJ_param ← clip(LJ_param, 0.001, 100.0)
```

This is a **hard** wall — parameters never leave `[0.001, 100.0]` kJ/mol
(or nm for sigma).  No tuning needed; it is always on.

> **What you see in the log**: when the 4th parameter (epsilon_Zn2+)
> reaches the lower bound and stays at `1.0000e-03` every epoch, the
> optimizer is trying to push it toward zero but the projection stops it.
> This is expected — it means that parameter is physically at its lower
> bound for this system.

### 2. Soft harmonic constraint (optional, set in `training.toml`)

```toml
[nn.loss]
constraint    = "harmonic"   # "harmonic" | "cubic" | null/omit
k_constraint  = 0.01         # spring constant  (kJ/mol / unit²)
boundary      = 50           # sigmoid upper wall (optional, kJ/mol)
```

The constraint adds a penalty term to the scalar loss:

$$
\mathcal{L}_\text{constraint} = k \sum_{i} (\theta_i - \theta_i^0)^2
$$

where $\theta_i^0$ are reference values defined in `[nn.model].LJ_type_param`
(for type-mode) or via `epsl_constraints` in the model (for pair-mode).

**When to use which:**

| Situation | Recommendation |
|---|---|
| Parameters drift wildly in epoch 0 | Increase `k_constraint` (e.g. 0.1 → 1.0) |
| Loss decreases but distance barely moves | Decrease `k_constraint` (0.01 → 0.001) to give optimizer more freedom |
| A parameter hits the hard wall and stays there | The physics wants it there; consider whether the target is reachable with that parameter fixed at the bound |
| Parameters oscillate without converging | Use `constraint = "cubic"` for a sharper (but less smooth) restraint |

**Effect on your current run:**

Your `k_constraint = 0.01` is very soft — it barely restrains the
parameters.  The dominant effect is the hard projection at 0.001.
The 4th parameter (epsilon Zn2+) has been clamped at the lower bound
since epoch 1 and is not contributing to the distance reduction.

**Suggested tuning workflow:**

1. Start with `k_constraint = 0.0` (no penalty) and wide bounds to see
   natural parameter directions.
2. Add `k_constraint = 0.01` if any parameter shows instability.
3. If a specific parameter needs to stay near a known value, add it
   explicitly via `epsl_constraints` in the model definition.

### 3. Boundary sigmoid wall (optional)

```toml
[nn.loss]
boundary   = 50     # upper energy wall (kJ/mol) — uses sigmoid penalty
boundary_S = 2      # steepness of the sigmoid
boundary_C = 500    # amplitude of the penalty
```

This applies a smooth sigmoid penalty when epsilon approaches `boundary`:

$$
\mathcal{L}_\text{boundary} = \frac{C}{2} \sum_i \sigma\!\left((ε_i - B) \cdot S\right)
$$

Useful to prevent epsilon from drifting to unrealistically large values
during unconstrained exploration.  Leave unset (or `boundary = null`) if
the hard box projection at 100 kJ/mol is sufficient.

---


## Units for TOML inputs

Diff-MD uses a GROMACS-like convention:

- Length: `nm`
- Energy: `kJ/mol`
- Temperature: `K`
- Pressure: `bar`
- Time: `ps` (and converted to `ns` in performance logs)

### `options.toml` fields

- `n_steps`, `n_print`, `n_flush`, `respa_inner`, `ns_nlist`, `nrexcl`, `n_b`: unitless counts
- `time_step`, `tau` (`tau_t` internally), `tau_p`: `ps`
- `start_temperature`, `target_temperature`: `K`
- `thermostat`: string selector (`"v-rescale"`/`"csvr"` or `"no"`)
- `target_pressure`: `bar`
- `beta`: `bar^-1`
- `rv`, `rc`, `rlj`, `skin`: `nm`
- `sigma`: `nm` (numeric) or `"auto"` for PME auto-tuning from `rc` and `ewald_rtol`
- `ewald_rtol`: dimensionless target for PME real-space tail (default `1e-5`, used when `sigma = "auto"`)
- `mesh_size`: grid counts (unitless integers)
- `pme_order`: B-spline interpolation order for PME (integer, 2–8, default `2`). Order 2 = CIC (Cloud-In-Cell), 4 = cubic B-spline (GROMACS default).
- `dielectric_const`, `epsilon_rf`, `coulomb14_scale`, `lj14_scale`: dimensionless
- `unwrap_output`: boolean (`true`/`false`, default `true`). Controls whole-molecule unwrapping in the output trajectory (see below).
- `lj_input_source`: LJ source selector (`"auto"`, `"mixing"`, `"input"`)
- `LJ_type_param = [[type_i, sigma_i, epsilon_i], ...]`: per-type LJ parameters; pair parameters are mixed internally using `combining_rule`
- `LJ_pair_param = [[type_i, type_j, sigma, epsilon], ...]`: explicit pair table input (`sigma` in `nm`, `epsilon` in `kJ/mol`)
- `LJ_param = [[type_i, type_j, sigma, epsilon], ...]`: legacy alias of explicit pair table input (still accepted)

LJ input modes:

- `lj_input_source = "mixing"`: requires `LJ_type_param`, builds pair terms internally from `combining_rule`
- `lj_input_source = "input"`: requires explicit pair entries from `LJ_pair_param` (or legacy `LJ_param`)
- `lj_input_source = "auto"` (default): backward-compatible behavior based on available fields and `combining_rule`

## PME/Ewald Accuracy (`sigma` and `rc`)

For PME runs (`coulombtype = "pme"`), electrostatic accuracy depends strongly on
the Ewald split parameter `sigma` relative to the real-space cutoff `rc`.

Diff-MD uses real-space Ewald term:

$$
\phi_{\mathrm{real}}(r) \propto \frac{\mathrm{erfc}(\beta r)}{r},
\qquad
\beta = \sqrt{\alpha} = \frac{1}{\sqrt{2}\,\sigma}
$$

So a practical truncation indicator is:

$$
\mathrm{erfc}(\beta\,rc)
$$

If this value is too large, dense liquids can show noticeable Coulomb-energy
offsets versus GROMACS even when bonded and LJ terms match.

Recommended target:

- Keep `erfc(beta*rc) <= 1e-5` for GROMACS-like default PME accuracy.

Auto mode:

- Set `sigma = "auto"` to let Diff-MD compute the optimized `sigma` from
  `rc` and `ewald_rtol`.
- Default `ewald_rtol` is `1e-5`.

For `rc = 0.9` nm, this corresponds to:

- `sigma ~= 0.2038` nm (good match target)

For reference:

- `sigma = 0.2735` nm -> `erfc(beta*rc) ~= 1e-3`
- `sigma = 0.2313` nm -> `erfc(beta*rc) ~= 1e-4`
- `sigma = 0.2038` nm -> `erfc(beta*rc) ~= 1e-5`

Notes:

- Very large `sigma` values (e.g. `sigma = 0.5` with `rc = 0.9`) imply a large
  real-space truncation tail (`erfc(beta*rc) ~= 7e-2`), which can bias Coulomb
  energies in dense systems.
- Single-molecule vacuum-like checks can still look correct, because truncation
  error is dominated by intermolecular near-cutoff pairs.
- Diff-MD now emits a startup warning when PME is active and truncation is large,
  and reports a recommended `sigma` for the current `rc`.

Example (`options.toml`):

```toml
[simulation]
coulombtype = "pme"
rc = 0.9

[field]
sigma = "auto"
ewald_rtol = 1e-5
mesh_size = [24, 24, 24]
```

## B-spline PME Interpolation

Diff-MD supports higher-order B-spline charge interpolation for the PME
reciprocal-space sum.  The interpolation order is controlled by the
`pme_order` option in `options.toml`.

### Background

In Particle Mesh Ewald (PME) the particle charges are spread ("painted")
onto a regular mesh, the Poisson equation is solved in Fourier space via
FFT, and the resulting potential is interpolated back to particle
positions.  The quality of the paint/readout step depends on the
interpolation function.

| `pme_order` | Name | Stencil (3-D) | Description |
|:-----------:|:----:|:-------------:|:------------|
| 2 | CIC (Cloud-In-Cell) | 8 pts | Linear interpolation (Diff-MD default) |
| 3 | TSC (Triangular-Shaped Cloud) | 27 pts | Quadratic B-spline |
| 4 | Cubic B-spline | 64 pts | GROMACS / AMBER default |
| 5–8 | Higher-order B-splines | $p^3$ pts | Rarely needed; useful for very coarse meshes |

Higher orders produce smoother charge distributions on the mesh, which
reduces aliasing artifacts and improves force-energy consistency,
especially on coarser meshes.

### Mathematical formulation

The cardinal B-spline of order $p$ is evaluated in closed form:

$$
M_p(x) = \frac{1}{(p-1)!}\sum_{k=0}^{p}(-1)^k\binom{p}{k}(x-k)_+^{p-1}
$$

where $(\cdot)_+ = \max(\cdot, 0)$.  The 3-D interpolation weight for
grid point $(i,j,k)$ relative to particle position $\mathbf{u}$ is the
product $M_p(u_x - i)\,M_p(u_y - j)\,M_p(u_z - k)$.

This closed-form expression is fully differentiable and compatible with
`jax.value_and_grad`, so forces are computed analytically (not by finite
difference).

### B-spline deconvolution (influence function correction)

Spreading charges with B-splines introduces a convolution with the
interpolation kernel in real space, which appears as a multiplicative
damping in Fourier space.  To recover the correct electrostatic potential,
Diff-MD applies an **inverse B-spline deconvolution** in the influence
function for orders $p > 2$.

For each mesh axis of size $M$ the deconvolution factor at frequency
index $m$ is:

$$
D(m) = \left[\frac{\sin(\pi m / M)}{\pi m / M}\right]^{-2p}
$$

with $D(0) = 1$.  The 3-D correction is the outer product of the
per-axis 1-D factors and is multiplied into the Gaussian window before
filtering the charge density in Fourier space.

This is the standard approach used by GROMACS, AMBER, and other PME
implementations to couple higher-order interpolation with the optimal
influence function (Essmann et al., J. Chem. Phys. 103, 8577, 1995).

**Impact on accuracy:** the deconvolution dramatically improves energy
convergence on coarse meshes.  Example relative errors for total PME
energy of a 4-water system (reference: order 4, mesh $32^3$):

| Mesh | CIC (order 2) | B-spline order 4 |
|:----:|:-------------:|:-----------------:|
| $8^3$ | 1.53 | 7.0 × 10⁻³ |
| $12^3$ | 1.36 | 1.3 × 10⁻³ |
| $24^3$ | 0.24 | 8.2 × 10⁻⁴ |
| $32^3$ | — | — (reference) |

Order 4 with deconvolution reaches sub-percent accuracy even at mesh
$8^3$, while CIC still has > 100 % error at mesh $12^3$.  The
deconvolution is applied automatically for `pme_order >= 3`; CIC
(`pme_order = 2`) is left unchanged for backward compatibility.

### TOML configuration

Add `pme_order` to your `options.toml`:

```toml
[simulation]
coulombtype = "pme"
rc = 0.9
pme_order = 4          # cubic B-spline (default: 2 = CIC)

[field]
sigma = "auto"
ewald_rtol = 1e-5
mesh_size = [24, 24, 24]
```

`pme_order` must be an integer between 2 and 8.  If omitted, it defaults
to 2 (CIC), preserving backward compatibility.

### Performance

Benchmark on a 4-water test system (CPU, 20 repetitions):

| Operation | CIC (order 2) | B-spline (order 4) | Ratio |
|-----------|:-------------:|:------------------:|:-----:|
| Paint | 0.078 ms | 0.095 ms | 1.22x |
| Full PME | 0.257 ms | 0.321 ms | 1.25x |

The overhead is modest because JAX's XLA compiler fuses the weight
computation and scatter/gather into efficient kernels.  On GPU the
relative cost is expected to be even smaller since the FFT dominates.

### When to use higher orders

- **Order 2 (CIC):** fast, sufficient when the mesh is fine relative to the
  system size (e.g. > 1 grid point per nm).
- **Order 4:** recommended for production runs that need GROMACS-level PME
  accuracy, especially with `sigma = "auto"` tuning.
- **Order 6–8:** useful for very coarse meshes or high-precision benchmarks.

## LJ Force-Shift and NVE Ensemble

### Force-shifted Lennard-Jones

Diff-MD implements a **force-shifted LJ** potential that ensures both the
energy and force go smoothly to zero at the cutoff $r_c$:

$$
V_{\text{shift}}(r) = \bigl[V_{\text{LJ}}(r) - V_{\text{LJ}}(r_c)\bigr]
                      - (r - r_c)\,\left.\frac{\mathrm{d}V_{\text{LJ}}}{\mathrm{d}r}\right|_{r_c}
$$

This eliminates the discontinuity in forces at the cutoff that is present in
a plain truncated or potential-shifted LJ.  A continuous force profile is
critical for energy conservation in NVE simulations.

Configuration:

```toml
[simulation]
lj_force_shift = true   # explicitly enable; auto-enabled for NVE
```

When `ensemble = "NVE"` is set, `lj_force_shift` is automatically turned on
(with an informational log message) unless it was already set explicitly.

### NVE ensemble

Diff-MD supports microcanonical (NVE) simulations.  When `ensemble = "NVE"`:

- The thermostat is disabled (`thermostat = "no"`).
- LJ force-shift is auto-enabled for energy conservation.
- The integrator is pure velocity Verlet (single time step).

Example NVE configuration:

```toml
[simulation]
ensemble = "NVE"
n_steps = 200000
time_step = 0.0005       # 0.5 fs — typical for NVE water
n_print = 50
rv = 1.1
rc = 0.9
rlj = 0.9
skin = 0.2
ns_nlist = 2
coulombtype = "pme"
cancel_com_momentum = 100

[field]
sigma = "auto"
ewald_rtol = 1e-5
mesh_size = [24, 24, 24]
```

### Supported ensembles

Only `"NVT"` (default) and `"NVE"` are supported.  Any other value raises
an error at startup.

### Double Precision (float64)

By default Diff-MD runs in single precision (float32).  Double precision
(float64) can be enabled in three ways, listed in order of precedence:

| Method | Scope | Where to set |
|--------|-------|--------------|
| `ensemble = "NVE"` | Automatic | `options.toml` — `[simulation]` section |
| `--double-precision` CLI flag | Per run | Command line (`mdrun` or `optimize`) |
| `double_precision = true` in TOML | Per training config | `training.toml` — `[nn]` section |

The precision flag is set *before* any JAX array is created (a JAX
requirement), by performing a lightweight early parse of the TOML config.

#### What gets promoted to float64

When double precision is active:

- **JAX global flag** `jax_enable_x64` is set to `True`.  All subsequent
  `jnp` operations use float64 by default.
- **Positions and velocities** loaded from the input H5 file are
  automatically upcast to float64 (even if the file stores float32).
- **PME mesh operations** (B-spline deconvolution factors, FFTs, influence
  function) run in float64.
- **Force and energy accumulation** (LJ, Coulomb, bonded) all use float64.
- **Optimizer state** (Optax adam / multi-step) operates on float64
  parameter arrays.

#### When to use double precision

- **NVE (microcanonical):** Mandatory.  Float32 accumulation errors cause
  visible energy drift; Diff-MD enables float64 automatically.
- **NVT optimization with JVP gradients:** Recommended.  Forward-mode
  gradients carry fewer cancellation errors in float64, giving
  near-machine-precision agreement with reverse-mode (~1e-7 relative
  error vs ~1e-3 in float32).
- **NVT optimization with reverse-mode:** Optional.  Float32 is usually
  sufficient for practical gradient-based parameter fitting.
- **Production NVT `mdrun`:** Rarely needed.  The thermostat absorbs small
  integration errors.

#### Enabling double precision

**1. NVE ensemble (automatic)**

```toml
# options.toml
[simulation]
ensemble = "NVE"    # ← float64 enabled automatically
```

**2. CLI flag (any ensemble, mdrun or optimize)**

```terminal
diff_md mdrun --double-precision -f input.h5 -p topol.toml -c options.toml -o sim -v

diff_md optimize --double-precision -f input.h5 -p topol.toml -c options.toml \
     -m training.toml -o train -d $PWD -v
```

**3. Training TOML (optimization only)**

```toml
# training.toml
[nn]
double_precision = true
n_epochs = 100
# ... other training options ...
```

The TOML setting and the CLI flag are equivalent.  If either is set, float64
is enabled.  This avoids having to remember the flag on every `srun` call.

#### Performance impact

Float64 roughly doubles memory usage and can reduce throughput by 1.5–2× on
consumer GPUs (which have limited float64 ALUs).  Data-center GPUs (A100,
H100, MI250X) have full-rate float64 units and show minimal slowdown.

#### Behaviour summary

- `ensemble = "NVE"` → `jax_enable_x64 = True` (with an info-level log).
- `--double-precision` flag → `jax_enable_x64 = True` (any ensemble).
- `double_precision = true` in `[nn]` → `jax_enable_x64 = True`.
- Otherwise → default float32.

## Integrator

Diff-MD uses a **velocity Verlet** integrator with a single time step:

1. Half-step velocity update: $v \leftarrow v + \tfrac{1}{2}\,\Delta t\,a$
2. Full position update: $x \leftarrow x + \Delta t\,v$
3. Recompute all forces at new positions
4. Second half-step velocity: $v \leftarrow v + \tfrac{1}{2}\,\Delta t\,a$
5. Apply thermostat (NVT) or identity (NVE)

The step body is JIT-compiled via `jax.jit` for performance.  All Python-level
branching (topology flags, coulomb type, 1-4 scaling, etc.) is resolved at
trace time through a factory closure, so the compiled kernel contains only the
code paths actually needed by the current system.

### Inner `lax.scan` loop

The MD loop uses `jax.lax.scan` to run *chunks* of steps entirely on-device
(GPU or CPU) without returning to Python on every step.  The outer Python loop
iterates over output frames (every `n_print` steps); within each frame the
inner `lax.scan` executes `n_print` velocity-Verlet steps — including neighbor
list updates and COM momentum cancellation — as a single fused XLA program.

Benefits:

- Eliminates per-step Python / host-launch overhead.
- Enables XLA to fuse and optimize across multiple time steps.
- Neighbor list overflow is checked once per output frame (not per step);
  for jax-md lists, the capacity is re-allocated between frames if needed.

## NPT Barostat (Constant Pressure)

Diff-MD supports constant-pressure (NPT) simulations with two barostat
algorithms: **Berendsen** and **C-Rescale** (stochastic cell rescaling).

### Enabling NPT in `options.toml`

Set the ensemble to `"NPT"` and add the barostat parameters:

```toml
[simulation]
ensemble    = "NPT"
n_steps     = 10000
time_step   = 0.0005
# ... thermostat settings (v-rescale recommended) ...

pressure         = true
barostat         = "scr"          # "berendsen" or "scr"
barostat_type    = "isotropic"    # "isotropic", "semiisotropic", or "surface_tension"
tau_p            = 2.0            # pressure coupling time constant (ps)
target_pressure  = [1.0, 1.0, 1.0]   # target pressure per axis (bar)
beta             = 4.5e-5         # isothermal compressibility (bar⁻¹)
```

#### Required fields

| Field | Type | Description |
|---|---|---|
| `ensemble` | string | Must be `"NPT"` |
| `pressure` | bool | Must be `true` |
| `barostat` | string | `"berendsen"`, `"scr"`, or `"no"` |
| `barostat_type` | string | `"isotropic"`, `"semiisotropic"`, or `"surface_tension"` |
| `tau_p` | float | Pressure coupling time constant in ps (typical: 1.0–5.0) |
| `target_pressure` | float or list | Target pressure in bar; scalar → same for all axes |
| `beta` | float | Isothermal compressibility in bar⁻¹ (default: 3.6×10⁻⁵) |

#### Optional fields

| Field | Default | Description |
|---|---|---|
| `n_b` | 1 | Barostat coupling frequency (applied every `n_b` steps) |

### Berendsen barostat

The Berendsen barostat rescales box vectors and positions by a factor:

$$\alpha = \sqrt[3]{1 - \frac{n_b \cdot \beta \cdot \Delta t}{\tau_p}\,(P_\text{target} - P_\text{inst})}$$

This drives the instantaneous pressure toward the target exponentially with
time constant $\tau_p$.  It is computationally simple and produces rapid
pressure equilibration, but it **does not generate the correct NPT ensemble**
— volume fluctuations are artificially suppressed.

**When to use:** Equilibration runs where correct fluctuations are not needed.

### C-Rescale (Stochastic Cell Rescaling) barostat

The C-Rescale (SCR) barostat adds a stochastic noise term to the Berendsen
scaling factor (Bernetti & Bussi, *J. Chem. Phys.* **153**, 114107, 2020):

$$\alpha = \exp\!\left(\frac{s + \eta}{3}\right)$$

where $s$ is the Berendsen drift term and $\eta$ is Gaussian noise with
variance:

$$\sigma^2 = \frac{2\,n_b\,\beta\,\Delta t\,k_BT\,p_\text{conv}}{V\,\tau_p}$$

The stochastic term ensures that the **correct NPT ensemble** is sampled,
meaning volume fluctuations yield the true isothermal compressibility:

$$\beta_T = \frac{\text{Var}(V)}{\langle V\rangle\,k_BT}$$

**When to use:** Production runs where correct thermodynamic averages and
fluctuation properties are required.

### Barostat types

- **Isotropic**: All box dimensions are scaled equally (cubic symmetry).
  Use `target_pressure = [P, P, P]` or a single scalar.
- **Semi-isotropic**: XY dimensions are coupled; Z is independent.
  Use `target_pressure = [Pxy, Pz]` (the code expands to `[Pxy, Pxy, Pz]`).
- **Surface tension**: Like semi-isotropic but XY pressure includes a
  surface-tension correction $\gamma / L_z$.  Only available with `"scr"`.

### Pressure output

When running NPT, the instantaneous pressure tensor (diagonal: $P_{xx}$,
$P_{yy}$, $P_{zz}$) is stored in the H5 output file under
`observables/pressure/value` (shape `(n_frames, 3)`, units: bar).

The scalar mean pressure is also printed in the `energy.log` file as the
`P_bar` column (average of the three diagonal components, in bar).

### Example: TIP3P water NPT simulation

```toml
[simulation]
ensemble = "NPT"
n_steps = 50000
n_print = 100
time_step = 0.0005

start_temperature = 300.0
target_temperature = 300.0
thermostat = "v-rescale"
tau = 0.5
thermostat_coupling_groups = [['OW', 'HW']]

pressure = true
barostat = "scr"
barostat_type = "isotropic"
tau_p = 2.0
target_pressure = [1.0, 1.0, 1.0]
beta = 4.5e-5

coulombtype = "pme"
rv = 1.1
rc = 0.9
rlj = 0.9
nrexcl = 3
```

## Neighbor List And Verlet Skin

Diff-MD now supports a displacement-based Verlet rebuild criterion with a configurable skin.

- The pair list is built with radius `rv`.
- Forces still use their own physical cutoffs (`rc`, `rlj`).
- A rebuild check is performed every `ns_nlist` steps.
- At each check step, the code computes the max particle displacement since the last rebuild (PBC minimum-image).
- Rebuild condition: `max_drift > skin / 2`.
- If the condition is false, the current list is reused and the next check happens after another `ns_nlist` steps.

Recommended starting values:

- `skin = 0.1` to `0.2` nm (1-2 Angstrom)
- threshold is then `0.05` to `0.1` nm (skin/2)

Important:

- For correctness, `rv` should satisfy `rv >= max(rc, rlj) + skin`.
- If `skin > 0` and `rv` is smaller, Diff-MD emits a warning and increases the effective runtime `rv` to `max(rc, rlj) + skin`.
- `skin = 0.0` keeps the previous behavior (rebuild on each `ns_nlist` check).

Example `options.toml` fragment:

```toml
[simulation]
rv = 1.1
rc = 0.9
rlj = 0.9
ns_nlist = 40
skin = 0.2
```

Performance note:

- Initial list build uses a cell-list implementation to reduce startup overhead vs brute-force O(N^2) pair enumeration.
- Runtime list rebuild remains JAX-compatible and traceable.

Verified run example (water TIP3P-FB test folder):

```terminal
cd atomistic_protein/water/tip3p-fb
source ../../../../test_vari/bin/activate
PYTHONPATH=../../../../src diff_md mdrun -f bo4.h5 -p topol.toml -c options.smoke.verlet.toml -o prova_verlet2 -v
```

This run completed to `60/60` steps and produced:

- `atomistic_protein/water/tip3p-fb/prova_verlet2.h5`
- `atomistic_protein/water/tip3p-fb/prova_verlet2.log`

Mixing rules (used when internal mixing is active):

- `combining_rule = "lorentz-berthelot"`: $\sigma_{ij} = (\sigma_i + \sigma_j)/2$, $\epsilon_{ij} = \sqrt{\epsilon_i\epsilon_j}$
- `combining_rule = "geometric"`: $\sigma_{ij} = \sqrt{\sigma_i\sigma_j}$, $\epsilon_{ij} = \sqrt{\epsilon_i\epsilon_j}$
- `combining_rule = "pairtable"`: uses explicit pair entries (`LJ_pair_param`/`LJ_param`)

Example: internal mixing from per-type LJ

```toml
[atomistic_ff]
combining_rule = "lorentz-berthelot"
lj_input_source = "mixing"

[field]
LJ_type_param = [
  ["OW", 0.3150, 0.6360],
  ["HW", 0.0000, 0.0000],
]
```

Example: explicit pair-table input

```toml
[atomistic_ff]
combining_rule = "pairtable"
lj_input_source = "input"

[field]
LJ_pair_param = [
  ["OW", "OW", 0.3150, 0.6360],
  ["OW", "HW", 0.1575, 0.0000],
  ["HW", "HW", 0.0000, 0.0000],
]
```

## Neighbor List Rebuild Logic (detailed)

Both backends (cell-list and jax-md) share the same two-level rebuild
strategy.  Understanding it is important for choosing good `ns_nlist`
and `skin` values.

### Decision tree (every MD step)

```
step % ns_nlist == 0 ?
├─ NO  → skip check entirely, keep current list
└─ YES → compute max single-particle displacement since last rebuild
         (PBC minimum-image, no sqrt — compared in squared form)
         ├─ max_disp² >= (skin / 2)²  OR  skin == 0 ?
         │   ├─ YES → rebuild list, store current positions as new reference
         │   └─ NO  → keep list, reference positions unchanged
```

The idea is the classical Verlet-list criterion: if no particle has
moved more than `skin / 2` since the last rebuild, the pair list still
contains every pair within the force cutoff plus some buffer.  Two
approaching particles that each drifted by at most `skin / 2` can
collectively close a gap of at most `skin`, which is exactly the buffer
that was included at build time (`rv >= max(rc, rlj) + skin`).

Setting `skin = 0` degenerates to an unconditional rebuild every
`ns_nlist` steps (backward-compatible behavior).

### What happens at rebuild time

1. **Cell-list path** (`nlist_method = "cell"`, default):
   - The *initial* build (before the simulation loop) uses a NumPy
     cell-list algorithm (`build_neighbor_list_cell`) that runs in
     $O(N)$ time.
   - *Runtime* rebuilds inside `lax.scan` use the JIT-compatible
     `nlist()` function, which is $O(N^2)$ (brute-force distance
     matrix + upper-triangle mask).  This is a known limitation:
     cell-list algorithms require dynamic shapes that are hard to
     express inside JAX tracing.  For typical system sizes
     (< 10 k atoms) and moderate `ns_nlist` the overhead is
     acceptable.
   - Bonded exclusions are applied via a hash-set
     (`exclude_bonded_neighbors`) with $O(n_{\text{pairs}} +
     n_{\text{excl}})$ cost.

2. **jax-md path** (`nlist_method = "jaxmd"`):
   - Uses jax-md's `OrderedSparse` cell-list which is $O(N)$ both at
     init *and* at runtime.
   - `nbrs.update(positions)` performs jax-md's own displacement
     check internally (same `skin / 2` threshold), so the rebuild
     is only triggered when truly needed.
   - Pair buffer overflow is detected via `did_buffer_overflow` and
     handled with a re-allocation in `mdrun`.

### Choosing `ns_nlist` and `skin`

- `ns_nlist` controls how often the displacement is *checked*, not how
  often the list is rebuilt.  Smaller values detect drift sooner but add
  per-check overhead.  Typical values: 10–100.
- `skin` controls how much buffer is included.  Larger skin → fewer
  rebuilds, but more pairs in the force evaluation.  Typical: 0.1–0.3 nm.
- For a first guess use `skin ≈ 0.1 * rv` and `ns_nlist = 20`.

## jax-md Neighbor List (alternative backend)

Diff-MD can optionally use [jax-md](https://github.com/google/jax-md)'s
O(N) cell-list-based neighbor search instead of the built-in Verlet list.
This is selected with `nlist_method = "jaxmd"` in `options.toml`.

### How it works

- jax-md builds an `OrderedSparse` neighbor list that returns unique
  (i < j) pairs, matching Diff-MD's half-shell convention.
- The physical cutoff passed to jax-md is `r_cutoff = max(rc, rlj)`.
- The Verlet skin is passed as `dr_threshold`, so the effective search
  radius is `max(rc, rlj) + skin`.
- **The `rv` parameter in TOML is not used by jax-md mode**.  The search
  radius is fully determined by `rc`, `rlj`, and `skin`.  For the common
  case where `rv = max(rc, rlj) + skin`, both methods produce the same
  search radius.
- Rebuilds are gated by `ns_nlist` (same as the cell-list path) and
  jax-md's internal displacement check against `skin / 2`.

### TOML configuration

```toml
[simulation]
nlist_method = "jaxmd"   # "cell" (default) or "jaxmd"
rv = 1.1                 # ignored by jaxmd, used only by cell-list path
rc = 0.9
rlj = 0.9
skin = 0.2
ns_nlist = 2
```

### A/B comparison: cell-list vs jax-md

Both methods were run on the same TIP3P-FB water system (1536 atoms,
box ≈ 2.49 nm, NVT v-rescale, PME electrostatics, 100 steps, dt = 0.002 ps)
starting from identical initial conditions.

| Metric | cell-list | jax-md | delta |
|---|---|---|---|
| E_mean (kJ/mol) | −20303 | −20262 | +41 |
| E_std  (kJ/mol) |    571 |    566 | −5 |
| T_mean (K) |  140.7 |  141.6 | +0.9 |
| T_std  (K) |   53.6 |   53.9 | +0.3 |
| \|P\|_mean |  5.2 × 10⁻⁵ |  5.7 × 10⁻⁵ | +0.5 × 10⁻⁵ |
| E_std ratio (jaxmd/cell) | — | — | 0.991 |
| Valid pairs | 427 616 | 427 616 | 0 |

Step-0 energies are bit-identical (all deltas = 0.0000).  Subsequent
divergence (O(40) kJ/mol over 100 steps) comes from different pair
ordering → different float32 force accumulation → different thermostat
random draws.  All physical observables are statistically equivalent.

### Performance (CPU-only, no GPU)

| Method | MD loop | step/s | Pair capacity |
|---|---|---|---|
| cell-list | 31.4 s | 3.25 | 427 616 |
| jax-md (optimized) | 34.1 s | 2.99 | 448 996 |

On CPU, jax-md is ~8% slower due to slightly larger pair arrays
(5% padding).  On GPU jax-md is expected to be faster for larger systems.

### Overflow handling (mdrun only)

If jax-md's pair buffer overflows at runtime, `mdrun` detects the
`did_buffer_overflow` flag and re-allocates with `neighbor_fn.allocate()`.
This is logged as a warning.  In the `simulate.py` (lax.scan) path,
overflow is accumulated on-device and reported after the scan completes.

## Testing

Diff-MD includes a lightweight pytest suite focused on physical sanity checks,
thermostat behavior, and neighbor-list correctness.

Run all tests:

```terminal
source test_vari/bin/activate
python -m pytest tests/ -v
```

### What the tests cover

- `tests/test_force_balance_water.py`
  - Checks net-force closure in a closed water box (`sum_i F_i ~ 0` each frame).
- `tests/test_virial_pressure_water.py`
  - Reconstructs pressure from kinetic + virial terms and checks finiteness/consistency.
- `tests/test_thermostat_math.py`
  - Verifies thermostat math edge cases (`thermostat = "no"`, finite v-rescale behavior).
- `tests/test_verlet_energy_conservation.py`
  - Validates Verlet-neighbor-list behavior in NVT by comparing:
    - `skin = 0.2` (Verlet)
    - `skin = 0.0` (baseline rebuild behavior)
  - Confirms comparable total-energy statistics, force closure, and COM-momentum stability.
- `tests/test_bspline_pme.py`
  - **B-spline weight properties:** weights partition unity and are non-negative (orders 2–6).
  - **CIC equivalence:** order-2 B-spline reproduces CIC paint, readout, and energy exactly.
  - **Force-energy consistency:** for orders 3, 4, 5 verifies that PME forces = $-\nabla E$ (numerical gradient check).
  - **Accuracy improvement:** order-4 paint→readout round-trip is closer to true charges than CIC; order-4 total PME energy on a coarse mesh ($12^3$) converges faster to the fine-mesh reference than CIC (verifies B-spline deconvolution).
  - **Performance benchmark:** CIC vs B-spline order-4 wall-clock comparison (informational).

### NVT energy note (important)

In NVT with `v-rescale`, total energy is not strictly conserved because the
thermostat exchanges heat with the system. For this reason, the Verlet test is
formulated as a **relative comparison** against a baseline run with identical
setup and `skin = 0.0`, rather than requiring flat total energy.

### Reproducing the Verlet NVT comparison (TIP3P-FB water)

From `atomistic_protein/water/tip3p-fb/`:

```terminal
source ../../../test_vari/bin/activate

# Verlet run (skin > 0)
diff_md mdrun -f bo4.h5 -p topol.toml -c options.econs_verlet.toml -o econs_verlet -v

# Baseline run (skin = 0)
diff_md mdrun -f bo4.h5 -p topol.toml -c options.econs_baseline.toml -o econs_baseline -v

# Then run pytest
cd ../../../
python -m pytest tests/test_verlet_energy_conservation.py -v
```

Expected outputs:

- `atomistic_protein/water/tip3p-fb/econs_verlet.h5`
- `atomistic_protein/water/tip3p-fb/econs_baseline.h5`

### Protein-scale manual check (Verlet rebuild + exclusions)

For a larger atomistic validation (protein + solvent), use:

```terminal
cd atomistic_protein
source ../test_vari/bin/activate
diff_md mdrun -f output.h5 -p topol.toml -c options.verlet.toml -o prova_verlet_protein -v
```

This is useful to inspect rebuild robustness and 1-X exclusion handling under
realistic system size and topology complexity.

### `topol.toml` bonded terms

- `bonds = [i, j, funct, r0, k]`
	- `r0`: `nm`
	- `k`: `kJ/mol/nm^2` (harmonic bond)
- `angles = [i, j, k, funct, theta0_deg, k]`
	- `theta0_deg`: degrees in file (converted to radians internally)
	- `k`: `kJ/mol/rad^2` (harmonic angle)
- Amber-like proper dihedrals: `[i, j, k, l, funct, phi0_deg, k, multiplicity]`
	- `phi0_deg`: degrees in file (converted to radians internally)
	- `k`: `kJ/mol`
	- `multiplicity`: unitless
- Amber-like protein style (GROMACS export compatible): `[i, j, k, l, funct, improper_bool, phi0_deg, k, multiplicity]`
	- `improper_bool = false`: treated as proper periodic torsion (typically `funct = 9`)
	- `improper_bool = true`: treated as improper harmonic torsion (typically `funct = 4`)
	- `phi0_deg`: degrees in file (converted to radians internally)
	- `k`: `kJ/mol` for proper periodic, `kJ/mol/rad^2` for improper harmonic
- Impropers (either in `dihedrals` with `funct = 2` or in `impropers` section)
	- `phi0_deg`: degrees in file (converted to radians internally)
	- `k`: `kJ/mol/rad^2` (harmonic improper)

### Notes

- `kappa`, `rho0`, and `a` are read from TOML but are not fully documented with explicit physical units in code comments.
- `rho0` and `a` default to `n_particles / volume`, so they behave as number-density-like quantities (`1/nm^3`).

## Whole-molecule unwrapping for visualisation output

### The problem

In periodic boundary conditions (PBC), atoms are stored modulo the box
length.  This means a water molecule sitting near a box edge can have its
oxygen at $x = 9.9$ nm and a hydrogen wrapped to $x = 0.1$ nm.  The atoms
are physically bonded at a distance of ~0.1 nm, but a molecular viewer
(VMD, PyMOL, …) draws a bond stretching across the entire box, making the
trajectory look wrong.

### How Diff-MD solves it

Diff-MD now applies **automatic bond-graph-based whole-molecule imaging** to
every frame **at output time**.  The internal simulation coordinates are
never modified — forces, neighbor lists, and energetics are completely
unaffected.  Only the copy of positions written to the H5MD trajectory file
is unwrapped.

The algorithm works as follows:

1. **Pre-computation (once, at simulation start).**  Using the bond arrays
   (`bonds_2_atom1`, `bonds_2_atom2`) and the per-atom molecule IDs, the
   code builds:
   - A mapping from each molecule ID to its list of atom indices.
   - A per-atom bond adjacency list.
   - Molecules are partitioned into *small* (≤ 3 atoms, e.g. water, ions)
     and *large* (> 3 atoms, e.g. proteins, lipids).

2. **Per-frame unwrapping (every `n_print` steps).**
   - **Small molecules** (water, ions): handled with a **vectorised NumPy
     path**.  The first atom of each molecule is the anchor; every other
     atom is shifted to its minimum-image position relative to the anchor
     in a single array operation.
   - **Large molecules** (protein, lipids): handled with a **BFS traversal**
     through the bond graph.  Starting from the first atom (anchor), each
     bonded neighbour is placed at its minimum-image position relative to
     its parent in the traversal tree.  This correctly handles arbitrarily
     branched topologies.

### Performance

The unwrapping does **not** slow down the simulation in any meaningful way:

| Component | Cost | When |
|---|---|---|
| Pre-computation | ~15 ms (22 k atoms) | Once at startup |
| Per-frame unwrap | ~3 ms (22 k atoms, 7 k water + 1 protein) | Every `n_print` steps |
| MD force step | 100–1000+ ms | Every step |

Since `n_print` is typically 100–1000, the unwrapping overhead is
$< 0.003\%$ of total simulation wall time.

### Configuration

Unwrapping is **enabled by default**.  To disable it (and get raw
PBC-wrapped coordinates in the output), add to your `options.toml`:

```toml
unwrap_output = false
```

### When is it applied?

- **During the simulation** — automatically, at every output frame.  You do
  **not** need to run any post-processing script.
- The H5MD file written by `diff_md mdrun` will contain unwrapped positions
  that can be loaded directly into VMD, PyMOL, or any other viewer without
  broken molecules.
- The dynamics themselves are **not** affected: forces, energies, neighbor
  lists, thermostats — everything still uses the normal PBC positions.

### Post-analysis usage

If you have an H5MD file that was written *without* unwrapping (e.g. from
an older version, or with `unwrap_output = false`), you can unwrap it in a
Python script:

```python
import h5py
import numpy as np
from diff_md.file_io import _build_unwrap_data, unwrap_molecules

with h5py.File("trajectory.h5", "r") as f:
    positions = np.array(f["particles/all/position/value"])     # (n_frames, n_atoms, 3)
    box_edges = np.array(f["particles/all/box/edges/value"])    # (n_frames, 3, 3)
    molecules = np.array(f["parameters/vmd_structure/resid"])   # (n_atoms,)
    bond_from = np.array(f["parameters/vmd_structure/bond_from"]) - 1  # 0-based
    bond_to   = np.array(f["parameters/vmd_structure/bond_to"])   - 1

sa, so, lma, adj = _build_unwrap_data(molecules, bond_from, bond_to)

unwrapped = np.empty_like(positions)
for frame in range(len(positions)):
    box_size = np.diag(box_edges[frame])
    unwrapped[frame] = unwrap_molecules(
        positions[frame], box_size, sa, so, lma, adj,
    )
# unwrapped now contains whole-molecule positions for every frame
```

### Technical details

- The anchor atom of each molecule is its first atom in index order.
  This atom keeps its original (possibly wrapped) coordinate; all other
  atoms are shifted relative to it.  Atoms may therefore end up slightly
  outside `[0, box)` — this is intentional and correct for visualisation.
- Single atoms (ions without bonds, lone particles) are left untouched.
- The implementation lives in `src/diff_md/file_io.py`
  (`_build_unwrap_data`, `unwrap_molecules`).



## Optimization call graph

```
optimize.py: value_and_grad(loss_fn)
  → losses.py: loss_fn calls simulator()
    → simulate.py: lax.scan outer loop
      → force.py, nonbonded.py, integrator.py, thermostat.py  (all @jit)
  → losses.py: post-processes trj dict → returns scalar error
← grads flow back through the whole chain
```

Custom loss function template:

```python
def my_custom_loss(model, system, key, start_temperature, comm,
                   <your_args>, metric, <optional_args>):
    sgm_table, epsl_table, _, types = get_LJ_param(
        model, system.config, jnp.array(system.types)
    )
    trj, key, config = simulator(
        model, system.positions, system.velocities, types,
        system.masses, system.charges, sgm_table, epsl_table,
        key, system.topol, system.config, start_temperature,
    )

    # Compute scalar error from trj using only jnp ops
    error = ...  # must be a JAX scalar

    return error, ({"my_observable": value}, trj, key, config, types)
```

## Available loss functions

All loss functions live in `src/diff_md/losses.py` and are selected by name
in the training TOML via `[nn.loss] name = "..."`.

### `density_and_apl`

Loss for lipid membranes.  Matches lateral density profile and area per lipid.

```toml
[nn.loss]
name = "density_and_apl"
metric = "mse"
density_weight = 1.0
apl_weight = 1.0
width_ratio = 1.0

[nn.system_args."my_system"]
com_type = "C1"        # atom name used to find the membrane centre of mass
n_lipids = 64
target_apl = 0.633     # nm² — target area per lipid
target_density = "reference_aa.xvg"   # two-column XVG: z (nm), density
```

### `radius_of_gyration`

Matches the time-averaged radius of gyration of a polymer / protein chain.

```toml
[nn.loss]
name = "radius_of_gyration"
metric = "mse"
rg_weight = 1.0

[nn.system_args."my_system"]
resname = "LIG"    # residue name of the molecule to track
n_chains = 1
target_rg = 0.45   # nm
```

### `radius_of_gyration_dist`

Matches the full Rg probability distribution (KDE) instead of just the mean.

```toml
[nn.loss]
name = "radius_of_gyration_dist"
metric = "mse"
rg_weight = 1.0
width_ratio = 1.0      # KDE bandwidth factor (1.0 = one bin width)

[nn.system_args."my_system"]
resname = "LIG"
n_chains = 1
target_dist = "reference_rg_dist.xvg"   # col 0 = bin centres (nm), col 1 = distribution
```

### `radius_of_gyration_and_end_to_end`

Joint loss on both Rg and end-to-end distance.

```toml
[nn.loss]
name = "radius_of_gyration_and_end_to_end"
metric = "mse"
rg_weight = 10.0
end_to_end_weight = 1.0

[nn.system_args."my_system"]
resname = "LIG"
n_chains = 1
target_rg = 0.45
target_end_to_end = 1.20   # nm
```

---

### `coordination_distance_dist`

Matches the pairwise distance distribution between atom groups (e.g. for zinc
coordination chemistry).  For each pair `(A, B)` the code collects **all
Na × Nb minimum-image distances** across the full trajectory, builds a KDE,
and compares it to the reference distribution.

KDEs are averaged across MPI ranks.  The loss is normalized by the number of
pairs so that adding more pairs does not change the magnitude.

```toml
[nn.loss]
name = "coordination_distance_dist"
metric = "mse"
dist_weight = 1.0
width_ratio = 1.0          # KDE bandwidth factor (1.0 = one bin width)
constraint = "harmonic"    # optional
k_constraint = 0.01
boundary = 50              # optional epsilon upper bound

[nn.system_args."my_system"]
# Each sub-list is [atom_name_A, atom_name_B].
# Atom names are matched against system.names (the per-atom name field in H5).
coord_pairs = [
    ["SZ", "Zn"],   # CYS sulfur  – Zn
    ["NZ", "Zn"],   # HIS nitrogen – Zn
    ["SD", "Zn"],   # MET sulfur  – Zn
]
# Reference file: col 0 = bin centres (nm), cols 1..N = distribution per pair.
# The order of columns must match the order of coord_pairs above.
target_coord_dist = "reference_coord_dist.xvg"
```

**Reference XVG format** (one column per pair + bin centres):
```
# distance  SZ-Zn    NZ-Zn    SD-Zn
0.18        0.0      0.0      0.0
0.20        0.05     0.01     0.04
...
0.35        0.0      0.0      0.0
```

Adding a new coordination pair requires only extending `coord_pairs` in the
TOML and adding the corresponding column to the reference XVG — no code
changes needed.

---

### `coordination_distance`

Simpler variant: matches the time-averaged mean distance for each pair.
Use this when you only care about peak distances, not full distributions.

```toml
[nn.loss]
name = "coordination_distance"
metric = "mse"
dist_weight = 1.0

[nn.system_args."my_system"]
coord_pairs = [
    ["SZ", "Zn"],
    ["NZ", "Zn"],
    ["SD", "Zn"],
]
# Target mean distance (nm) for each pair, in the same order as coord_pairs.
target_distances = [0.232, 0.210, 0.232]
```

The metric is applied directly to the vector of mean distances:
`loss = dist_weight * metric(mean_d_sim, target_distances)`.
With `metric = "mse"` this is the average squared deviation over all pairs.

---

### `coordination_tetrahedral`

Joint loss on **mean coordination distances** and the **orientational tetrahedral
order parameter** *q* for metal coordination sites (e.g., Zn²⁺ in
metalloproteins).

#### Mathematical background — tetrahedral order parameter *q*

The orientational tetrahedral order parameter was first proposed by
Chau & Hardwick (1998) and subsequently rescaled by Errington & Debenedetti
(2001) to vary between 0 (ideal gas / random arrangement) and 1 (perfect
regular tetrahedron).  It is computed over the six unique ligand-pair angles at the metal centre:

```
q = 1 - (3/8) * SUM_{all 6 pairs j<k} ( cos(psi_jk) + 1/3 )^2
```

Here `psi_jk` is the angle formed *at the metal atom* between the two
vectors pointing from the metal to ligands `j` and `k`.  With four
ligands there are C(4,2) = 6 such pairs.

**Physical interpretation.** In a perfect tetrahedron every such angle is
109.47° (= arccos(-1/3)), so every term in the sum is exactly zero and
q = 1.  Any distortion from tetrahedral symmetry pushes q downward:

| Geometry | q |
|---|---|
| Perfect tetrahedron             | 1.000      |
| Trigonal pyramid (one apex)     | ≈ 0.85     |
| Square planar                   | 0.500      |
| Random (ideal gas)              | ≈ 0        |

#### How sites are defined

Each coordination site is specified by **absolute atom indices** from the
H5 input file:

```
coord_sites = [
    [metal_idx, lig1_idx, lig2_idx, lig3_idx, lig4_idx],
    ...
]
```

This explicit definition lets you handle:

- **Bridging ligands** (same atom appears in multiple sites)
- **Mixed ligand types** (e.g. 2 Cys sulfurs + 2 His nitrogens around one Zn)
- **Multiple metal centres** (2, 3, or more Zn/Fe/Cu sites — unlimited)

#### Ligand grouping

Within each site the four ligand atoms are grouped by their atom name (as
stored in the H5 file).  Each group produces a separate target distance.
The column ordering — printed at startup — is **site-major, type-minor
(first-occurrence order within the site)**:

Example with 2 sites (site 0: 2×SZ + 2×NE2; site 1: 4×SZ):

```
Distribution group 0: site0 SZ-Zn   (site-local ligand slots 0-1, target_site_distances[0])
Distribution group 1: site0 NE2-Zn  (site-local ligand slots 2-3, target_site_distances[1])
Distribution group 2: site1 SZ-Zn   (site-local ligand slots 0-3, target_site_distances[2])
```

`target_site_distances` must then have 3 entries, in this order.

#### Per-site target *q*

Target *q* can be set globally (scalar — same for all sites) or per-site
(array of length `n_sites`):

```toml
# Same target for all sites
target_q = 1.0

# Different targets per site (e.g. one is distorted)
target_q = [1.0, 0.85]
```

#### Total loss

```
L = q_weight    * metric( mean_q_sim,    target_q    )     # geometry term
  + dist_weight * metric( mean_dist_sim, target_dist )     # distance term
```

`mean_q_sim` and `mean_dist_sim` are time-averaged (and MPI-averaged)
over the entire simulated trajectory.

#### TOML configuration

```toml
[nn.loss]
name = "coordination_tetrahedral"
metric = "mse"
dist_weight = 1.0
q_weight = 5.0
constraint = "harmonic"    # optional
k_constraint = 0.01
boundary = 50              # optional epsilon upper bound

[nn.system_args."my_system"]
# Each row: [metal_idx, lig1_idx, lig2_idx, lig3_idx, lig4_idx]
# Atom indices are absolute (0-based) from the H5 input file.
coord_sites = [
    [142, 10, 25, 40, 120],    # Zn site 0 — 4 cysteines (120 = bridging CYD)
    [143, 55, 70, 85, 120],    # Zn site 1 — 4 cysteines (120 shared)
]
# Target q per site (scalar or list).  1.0 = perfect tetrahedron.
target_q = [1.0, 1.0]
# Target mean distances per (site, ligand_type) group.
# Order must match the startup log (site-major, type-minor).
target_site_distances = [0.232, 0.232]
```

#### Practical example — two bridged Zn²⁺ sites

Consider a metalloprotein with two Zn²⁺ ions.  Site 0 is coordinated
by SZ atoms at indices 10, 25, 40 and a shared CYD sulfur (SD) at
index 120.  Site 1 is coordinated by NE2 at 55, SZ at 70, 85 and
the same bridging atom 120.

```toml
coord_sites = [
    [142, 10, 25, 40, 120],    #  Zn0: 3×SZ + 1×SD
    [143, 55, 70, 85, 120],    #  Zn1: 1×NE2 + 2×SZ + 1×SD
]
target_q = [1.0, 0.90]         # site 1 is somewhat distorted
target_site_distances = [
    0.232,    # site0 SZ-Zn
    0.235,    # site0 SD-Zn
    0.210,    # site1 NE2-Zn
    0.232,    # site1 SZ-Zn
    0.235,    # site1 SD-Zn
]
```

Run with `--debug` to verify which groups were assigned and check that
`target_site_distances` has the correct number of entries.

---

### `coordination_tetrahedral_dist`

Distribution-based variant of `coordination_tetrahedral`.  Uses KDE
matching for the per-group metal–ligand distances (like
`coordination_distance_dist`) while also enforcing the tetrahedral order
parameter *q*.

The *q* component always matches the mean *q* per site.  Optionally, a
full *q* distribution can also be matched.

#### Total loss

```
L = dist_weight * (1/N_groups) * SUM_g  metric( KDE_sim[g],   KDE_ref[g],   axis=bins )
  + q_weight* metric( mean_q_sim, target_q)
```

If `target_q_dist` is also provided, the following is added:

```
  + q_dist_weight * (1/N_sites) * SUM_s  metric( q_KDE_sim[s], q_KDE_ref[s], axis=bins )
```

`KDE_sim[g]` is the kernel-density estimate of the distance distribution for
group `g` built from all frames of this rank's trajectory; `q_KDE_sim[s]` is
the same for the per-site q time series.  Both are averaged across MPI ranks
before the metric is applied.

#### TOML configuration

```toml
[nn.loss]
name = "coordination_tetrahedral_dist"
metric = "mse"
dist_weight = 1.0
q_weight = 5.0
width_ratio = 1.0          # KDE bandwidth factor
q_dist_weight = 2.0        # weight for optional q distribution KDE term

[nn.system_args."my_system"]
coord_sites = [
    [142, 10, 25, 40, 120],
    [143, 55, 70, 85, 120],
]
target_q = [1.0, 1.0]
# Distance distribution: col 0 = bin centres (nm), cols 1..N = per group
target_coord_dist = "coord_dist_QMMM.xvg"
# Optional q distribution: col 0 = bin centres, cols 1..N_sites = per site
target_q_dist = "q_dist_QMMM.xvg"
```

#### Reference XVG file format — distance distributions

Same as `coordination_distance_dist`, but columns follow the per-site
per-type group ordering printed at startup.  Example for 2 all-SD sites:

```
# col 0: r (nm)   col 1: site0 SD-Zn   col 2: site1 SD-Zn
0.180    0.000    0.000
0.185    0.012    0.009
0.230    8.102    7.834
0.280    0.018    0.021
```

#### Reference XVG file format — *q* distributions

```
# col 0: q bin centre   col 1: site 0   col 2: site 1
0.50    0.001    0.002
0.60    0.010    0.015
0.70    0.045    0.060
0.80    0.180    0.210
0.90    0.520    0.480
0.95    0.780    0.650
1.00    0.320    0.290
```

---

### Common optional parameters (all losses)

| Parameter | Default | Description |
|---|---|---|
| `constraint` | `null` | Parameter constraint: `"harmonic"` or `"cubic"` |
| `k_constraint` | `0.01` | Constraint stiffness |
| `boundary` | `null` | Upper bound on ε_ij (kJ/mol); penalised via sigmoid |
| `boundary_S` | `2` | Sigmoid steepness for boundary penalty |
| `boundary_C` | `500` | Sigmoid amplitude for boundary penalty |

---

###Converting .itp and .top GROMACS structures 
```bash
# From a directory that contains the .gro/.pdb, .top, and amber19sb.ff/
python -m gmx2HyMD \
    -f input.gro \
    -p topol.top \
    -oc output.h5 \
    -op topol.toml \
```

2. Run `mdrun` or `optimize` exactly as usual; the CMAP energy is folded
   into the dihedral channel and surfaced as a separate `E_cmap_kJmol`
   column in `energy.log`.

```bash
diff_md mdrun -i input.h5 -m options.toml -p topol.toml -d run_out
```

`options.toml` after gmx2HyMD now ends with a block like:

```toml
[atomistic_ff]
ff_family = "amber-like"
combining_rule = "lorentz-berthelot"
...


### Optimizer warning (training only)

Plain SGD against `grad_method = "jvp"` on stochastic distribution losses
(Rg, density, coordination) is empirically unstable: a single noisy
gradient step can pin σ to `clip_sigma_max` in one epoch, after which the
next-epoch simulation collapses geometrically and jaxmd's neighbor list
overflows int32.  Mitigations, in order of preference:

1. Use `optimizer = "adabelief"` or `"adam"` — both have second-moment
   damping that absorbs the JVP noise.
2. If SGD is required, chain `optax.clip_by_global_norm(0.1)` before the
   SGD transform and drop the learning rate by 10–100×.
3. Tighten the clipping bounds: `clip_sigma_max ≈ 0.6` (or `1.5×` your
   largest physical σ), `clip_epsilon_min ≈ 0.05`.
4. Use the `boundary_constraint` term in the loss to apply a soft penalty
   instead of a hard post-update clip.

## Recent Changes

### CMAP backbone correction (`ff_family = "amber19sb"`)

Adds the standard ff19SB CMAP correction.  Per-residue 24×24 grids are
shipped in `options.toml`'s `[cmap]` block; per-chain backbone atom
indices live in each `Protein_chain_*.toml`.  Forces come from JAX
autodiff over a precomputed bicubic-Hermite spline.  See "AMBER ff19SB
/ CMAP Support" above for usage.  Tests: `tests/test_cmap.py` (13 cases).

### Coordination distance loss functions

Two new loss functions for metal coordination chemistry:

- **`coordination_distance_dist`** — matches Rdf-like distributions of
  pairwise distances between named atom groups (e.g. SZ–Zn, NZ–Zn, SD–Zn)
  against QM/MM reference distributions from `.xvg` files.  Internally builds
  a full-trajectory KDE per coordination pair, then averages across MPI ranks.
- **`coordination_distance`** — simpler version that matches only the
  time-averaged mean distance per pair.

Both functions are general: they handle an arbitrary number of zinc atoms and
an arbitrary number of coordination pairs.  Adding a new pair type requires
only extending `coord_pairs` in the TOML.

### MPI startup topology table

On startup `diff_md optimize` now prints a per-rank GPU assignment table:

```
MPI topology: 4 rank(s), backend=cuda
  Rank 0 -> CUDA:0 (CUDA_VISIBLE_DEVICES=0)
  Rank 1 -> CUDA:0 (CUDA_VISIBLE_DEVICES=1)
  Rank 2 -> CUDA:0 (CUDA_VISIBLE_DEVICES=2)
  Rank 3 -> CUDA:0 (CUDA_VISIBLE_DEVICES=3)
```

This replaces the previous debug-level rank-0-only log line, making it easy
to verify GPU pinning at a glance before a long training run.

### PME force-energy consistency fix

The reciprocal-space Coulomb forces are now computed as the exact negative
gradient of the reciprocal energy via `jax.value_and_grad`.  Previously,
forces were assembled by hand (spectral gradient on the mesh followed by
CIC readout), which did not commute with the CIC interpolation used for the
energy.  The mismatch caused a systematic force–energy inconsistency that
manifested as large NVE energy drift.

After the fix, NVE drift on a 4-water test system improved from
~8–25 kJ/mol/ps to ~0.33 kJ/mol/ps (~23× improvement).

The same approach was applied to `get_dipole_forces` for consistency.

### Force-energy consistency test suite

Five new tests (`tests/test_pme_force_energy_consistency.py`) verify that
forces are the exact gradient of the corresponding energy for:

- PME electrostatics (reciprocal + real space)
- Bonded interactions (bonds, angles, dihedrals, impropers)
- Lennard-Jones
- Full system (all terms combined)
- Net force magnitude (Newton's third law / momentum conservation)

### H5 restart from old trajectory files

Restarting from H5MD output files produced by older versions of Diff-MD
(before flat restart datasets were added) now works.  The parser
reconstructs `indices`, `types`, `names`, `molecules`, and `masses` from
the H5MD structure (`particles/all/species`, `parameters/vmd_structure/`,
etc.) when the flat datasets are absent.

Placeholder charges (e.g. all 1.0 from old files) are detected and
discarded with a warning.  A log message is emitted when restarting from
an existing trajectory.

### JAX index dtype fix

Scatter/gather operations that previously used `int64` indices (triggering
JAX `FutureWarning`) now use explicit `int32` dtype at creation and at
index-use boundaries.

### Exact restart with `--append`

The `--append` flag enables exact continuation of an interrupted or
completed `mdrun` simulation.  When `--append` is passed and the output
H5 file already exists, Diff-MD:

1. Opens the existing H5 file in append mode (no data is lost).
2. Reads the last written step/frame from the trajectory.
3. Resizes all time-dependent datasets for the new frames.
4. Offsets step numbers so the new run continues from the previous last
   step — step and time counters are fully continuous.
5. Appends energy.log entries without re-writing the header.

Input coordinates (`-f`) should point to the output trajectory from the
previous run (the H5MD file contains both positions/velocities and the
flat restart datasets needed to reconstruct the topology).

Usage example:

```terminal
# Initial run (100 steps)
diff_md mdrun -f input.h5 -p topol.toml -c options.toml -o sim -v

# Continue from where it stopped (another 100 steps)
diff_md mdrun -f sim.h5 -p topol.toml -c options.toml -o sim --append -v
```

After the second run, `sim.h5` will contain frames from step 0 through
step 200 with continuous step/time metadata.  The `energy.log` file is
likewise appended.

If `--append` is passed but the output file does not exist, a fresh
simulation is started (equivalent to not using `--append`).

Note: positions and velocities are automatically upcast to float64 when
restarting under NVE ensemble (or with `--double-precision`), even if the
previous run stored float32 output.



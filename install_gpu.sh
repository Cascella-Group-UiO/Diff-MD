#!/bin/bash
#prima $PATH deve essere resettato, no moduli
#module purge
#module load NRIS/GPU
#module load hpc-container-wrapper
#COMANDI UTILI, Compilazione va lancaita su nodo (GPU)

#CREA Container x OLIVIA
#conda-containerize new --prefix ./diff_amd_gpu prova.yml --post-install install_gpu.sh
#conda-containerize update ./diff_amd_gpu --post-install install_gpu.sh
#export PATH="/PATHTO/diff_amd_gpu/bin:$PATH"

#X lanciare su nodo CPUs
#salloc --ntasks=1 --cpus-per-task=128 --threads-per-core=1 --mem=0 --time=01:00:00 --account=??
#X lanciare su nodo GPUs
#salloc --ntasks=1 --threads-per-core=1 --cpus-per-task=128 --gpus=1 --time=01:00:00 --account=??? --partition=accel --mem 96G

#conda-containerize new --prefix ./diff_amd_gpu prova.yml --post-install install_gpu.sh

# 1. Update Pip
pip install --upgrade pip

# 2. Install JAX + CUDA 12 first
# We add --ignore-installed just in case something sneaky is left over
pip install -U "jax[cuda12]==0.9.1" --ignore-installed

# 3. Install the JAX Ecosystem
# We move these here so Conda doesn't trigger a JAX install
pip install flax optax chex jax-md dm-haiku vesin
pip install --no-cache-dir --force-reinstall uvloop
pip install --no-cache-dir --force-reinstall tensorstore

# 4. Utilities
pip install tomlkit toml simplejson treescope humanize

# 5. Build mpi4jax (The link)
export CUDA_ROOT=$CONDA_PREFIX
pip install nanobind
pip install --no-build-isolation mpi4jax

# 6. Local project
pip install --no-deps -e .

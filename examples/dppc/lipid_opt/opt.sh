#!/bin/bash
#SBATCH --account=nn4654k
#SBATCH --job-name=apl5.3
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=10
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=firmino.vinicius@usp.br

# Safety options, they stop the script if any error occours.
set -o errexit  # Recommended for easier debugging
set -o nounset  # Treat unset variables as errors

# module use ~/.local/easybuild/modules/all
# module load cuDNN/8.7.0.84-CUDA-11.8.0

module load Python/3.11.5-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0
source /cluster/projects/nn4654k/vfirmino/py_envs/new_diff_md/bin/activate

MPI4JAX_NO_WARN_JAX_VERSION=1

wd=`pwd`

srun -c 1 diff_md optimize -f input.h5 -p topol.toml -c options.toml -d $wd -v -m training/training.toml -o opt

exit 0



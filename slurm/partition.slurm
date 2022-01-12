#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Partitioning of images"
#SBATCH --output=partition.out
#SBATCH --mail-user=sebastvi@stud.ntnu.no
#SBATCH --mail-type=END

WORKDIR=${SLURM_SUBMIT_DIR}

cd ${WORKDIR} # /cluser/work/<username>/master-sau/slurm

uname -a
pip freeze --user | xargs pip uninstall -y
module purge
module load Python/3.8.6-GCCcore-10.2.0
# module load PyTorch/1.7.1-fosscuda-2020b

cd ..
cd preprocessing
pwd
pip install numpy opencv-python scikit-image --no-cache-dir

python -u transform.py




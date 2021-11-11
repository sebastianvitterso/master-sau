#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="debug.slurm"
#SBATCH --output=debug.out
#SBATCH --mail-user=sebastvi@stud.ntnu.no
#SBATCH --mail-type=NONE

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module spider PyTorch


uname -a

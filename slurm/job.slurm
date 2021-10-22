#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Testing_capabilities"
#SBATCH --output=job.out
#SBATCH --mail-user=sebastvi@stud.ntnu.no
#SBATCH --mail-type=NONE

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} # /cluser/work/<username>/master-sau/slurm

module purge
module load Python/3.8.6-GCCcore-10.2.0
# module load PyTorch/1.7.1-fosscuda-2020b

cd ..
cd yolov5
pwd
pip freeze --user | xargs pip uninstall -y
pip install -r requirements.txt --no-cache-dir

python train.py --img 1280 --batch 8 --epochs 3 --data sheep.yaml --weights yolov5s.pt --cache --freeze 10 --device 0
# python detect.py --weights yolov5s.pt --img 1024 --source ../data/test/

uname -a

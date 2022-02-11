#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100"
#SBATCH --account=ie-idi
#SBATCH --time=167:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="YOLOv5 Validation for sheep recognition"
#SBATCH --output=val.out

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} # /cluser/work/<username>/master-sau/slurm

uname -a

module purge
module load fosscuda/2020b
module load Python/3.8.6-GCCcore-10.2.0
# module load PyTorch/1.7.1-fosscuda-2020b

cd ..
cd yolov5
pwd
# # Unused pip installs, which need to be done before running
# pip install -r requirements.txt --no-cache-dir --user
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --user

# python detect.py --weights yolov5s.pt --img 1024 --source ../data/test/
FOLDER="fusion-no-msx"
python val.py --weights "runs/train/$FOLDER/weights/best.pt" --img 1280 --save-txt --save-conf --data sheep-cropped-no-msx.yaml --name $FOLDER


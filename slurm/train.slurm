#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100"
#SBATCH --account=ie-idi
#SBATCH --time=167:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="YOLOv5 Training for sheep recognition"
#SBATCH --output=train.out
#SBATCH --mail-user=sebastvi@stud.ntnu.no
#SBATCH --mail-type=END

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

# Enable wandb
wandb online
# wandb login

# # Disable wandb
# wandb disabled

# Train from pretrained weights
# python train.py --img 1280 --batch 6 --epochs 1000 --data sheep.yaml --weights yolov5l6.pt --cache --device 0
# python train.py --img 1280 --batch 6 --epochs 1000 --data sheep.yaml --weights '' --cfg yolov5l6.yaml --cache --device 0

# Train from scratch
# python train.py --img 1280 --batch 6 --epochs 1000 --data sheep.yaml --weights '' --cfg yolov5l6.yaml --cache --device 0

# Train from scratch with partitions
# python train.py --img 1280 --batch 6 --epochs 1000 --data sheep-partitioned.yaml --weights yolov5l6.pt --cache --device 0

# Train 4ch from scratch
python train.py --img 1280 --batch 6 --epochs 2000 --data sheep-cropped.yaml --weights '' --cfg yolov5l6.yaml --cache --device 0 --name 4ch

# Train 4ch from scratch with partitions
# python train.py --img 1280 --batch 6 --epochs 1000 --data sheep-cropped-partitioned.yaml --weights '' --cfg yolov5l6.yaml --cache --device 0



# python -m torch.distributed.launch --nproc_per_node 2 train.py --img 1280 --batch 8 --epochs 3 --data sheep.yaml --weights yolov5l6.pt --device 0,1
# python detect.py --weights yolov5s.pt --img 1024 --source ../data/test/
# python val.py --weights rgb.pt --img 1280 # --save-txt --save-conf 


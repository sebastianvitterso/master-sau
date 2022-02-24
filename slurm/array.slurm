#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100"
#SBATCH --account=ie-idi
#SBATCH --time=167:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Variance metrics"
#SBATCH --output=variance-train-3-%a.out
#SBATCH --mail-user=sebastvi@stud.ntnu.no,ingebrin@stud.ntnu.no
#SBATCH --mail-type=END
#SBATCH --array=1-50%10

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} # /cluster/work/<username>/master-sau/slurm

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

export ARRAY_RUN_NAME="variance_03_exp$(printf %02.0f $SLURM_ARRAY_TASK_ID)"

python train.py --img 1280 --batch 8 --epochs 2000 --data sheep-cropped-no-msx.yaml --weights '' --cfg yolov5l6.yaml --cache --device 0 --name "$ARRAY_RUN_NAME"

# python val.py --weights "runs/train/$ARRAY_RUN_NAME/weights/best.pt" --img 1280 --save-txt --save-conf --data sheep-cropped-no-msx.yaml --name "$ARRAY_RUN_NAME"


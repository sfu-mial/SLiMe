#!/bin/bash
#SBATCH --gres=gpu:a100:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to clusters documentation for the right CPU/GPU ratio
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:30:00     # DD-HH:MM:SS

module load python/3.6 cuda cudnn

OUTPUTDIR=/home/aka225/scratch/outputs
DATADIR=/home/aka225/scratch/data
OBJECTNAME=human
PARTNAME=nose

source /home/aka225/cps_env/bin/activate
cd ~/scratch/code/one_shot_segmentation
python3 -m src.main --base_dir $OUTPUTDIR \
                    --train \
                    --dataset celeba-hq \
                    --checkpoint_dir $OUTPUTDIR/"$OBJECTNAME"_"$PARTNAME"_te \
                    --objective_to_optimize text_embedding \
                    --part_names background $PARTNAME \
                    --images_dir $DATADIR/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img \
                    --masks_dir $DATADIR/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno \
                    --idx_mapping_file $DATADIR/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt \
                    --test_file_names_file_path $DATADIR/CelebAMask-HQ/CelebAMask-HQ/paper_test_file_names.txt \
                    --train_file_names_file_path $DATADIR/CelebAMask-HQ/CelebAMask-HQ/non_test_file_names.txt \
                    --val_file_names_file_path $DATADIR/CelebAMask-HQ/CelebAMask-HQ/non_test_file_names.txt \
                    --optimizer Adam \
                    --epochs 40 \
                    --self_attention_loss_coef 1 \
                    --lr 0.1 \
                    --crop_margin 10 \
                    --mask_size 128 \
                    --crop_threshold 0.2 \
                    --val_data_ids 200 201 202 203 204 205 206 207 208 209 \
                    --train_data_ids 0 \
                    # --train_data_ids 0 1 5 10 8 9 12 36 75 83 \

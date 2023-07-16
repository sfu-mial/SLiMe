#!/bin/bash
#SBATCH --gres=gpu:a100:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to clusters documentation for the right CPU/GPU ratio
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:20:00     # DD-HH:MM:SS

module load python/3.6 cuda cudnn

OUTPUTDIR=/home/aka225/scratch/outputs
DATADIR=/home/aka225/scratch/data
OBJECTNAME=car
PARTNAME=wheel
# Prepare virtualenv
source /home/aka225/cps_env/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.
cd ~/scratch/code/one_shot_segmentation

python3 -m src.main --base_dir $OUTPUTDIR \
                    --train \
                    --dataset pascal \
                    --checkpoint_dir $OUTPUTDIR/"$OBJECTNAME"_"$PARTNAME"_te \
                    --objective_to_optimize text_embedding \
                    --object_name $OBJECTNAME \
                    --part_names background $PARTNAME \
                    --train_data_file_ids_file $DATADIR/VOCdevkit/VOC2010/ImageSets/Main/"$OBJECTNAME"_train.txt \
                    --val_data_file_ids_file $DATADIR/VOCdevkit/VOC2010/ImageSets/Main/"$OBJECTNAME"_val.txt \
                    --remove_overlapping_objects \
                    --object_overlapping_threshold 0.05 \
                    --single_object \
                    --optimizer Adam \
                    --epochs 40 \
                    --self_attention_loss_coef 1 \
                    --lr 0.1 \
                    --mask_size 128 \
                    --crop_margin 100 \
                    --crop_threshold 0.2 \
                    --train_data_ids 0 1 3 4 5 6 7 8 9 10 \
                    --val_data_ids 12 13 14 17 19 20 21 22 29 31 \
                    # --fill_background_with_black

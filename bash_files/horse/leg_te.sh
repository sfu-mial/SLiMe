module load python/3.6 cuda cudnn

source /home/aka225/cps_env/bin/activate
cd ~/scratch/code/one_shot_segmentation

python3 -m src.main  --base_dir '/home/aka225/scratch/outputs' \
                    --train \
                    --dataset 'pascal' \
                    --checkpoint_dir '/home/aka225/scratch/outputs/horse_leg_te' \
                    --objective_to_optimize 'text_embedding' \
                    --object_name 'horse' \
                    --part_names 'background' 'leg' \
                    --train_data_file_ids_file '/home/aka225/scratch/data/VOCdevkit/VOC2010/ImageSets/Main/horse_train.txt' \
                    --val_data_file_ids_file '/home/aka225/scratch/data/VOCdevkit/VOC2010/ImageSets/Main/horse_val.txt' \
                    --remove_overlapping_objects \
                    --object_overlapping_threshold 0.05 \
                    --single_object \
                    --optimizer 'Adam' \
                    --epochs 40 \
                    --self_attention_loss_coef 1 \
                    --lr 0.1 \
                    --mask_size 128 \
                    --crop_margin 10 \
                    --crop_threshold 0.2 \
                    --train_data_ids 0 1 2 3 4 5 6 7 8 10 \
                    --val_data_ids 11 12 13 15 17 18 19 20 21 23 \
                    # --fill_background_with_black

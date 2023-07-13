module load python/3.6 cuda cudnn

source /home/aka225/cps_env/bin/activate

python3 -m src.main  --base_dir '/home/aka225/scratch/outputs' \
                    --train \
                    --dataset 'pascal' \
                    --checkpoint_dir '/home/aka225/scratch/outputs/horse_tail_tr' \
                    --objective_to_optimize 'translator' \
                    --object_name 'horse' \
                    --part_names 'background' 'tail' \
                    --train_data_file_ids_file 'self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/horse_train.txt' \
                    --val_data_file_ids_file 'self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/horse_val.txt' \
                    --remove_overlapping_objects \
                    --object_overlapping_threshold 0.05 \
                    --single_object \
                    --optimizer 'Adam' \
                    --epochs 40 \
                    --self_attention_loss_coef 1 \
                    --lr 0.001 \
                    --mask_size 128 \
                    --crop_margin 10 \
                    --crop_threshold 0.2 \
                    --train_data_ids 1 3 4 5 13 14 16 17 18 22 \
                    --val_data_ids 23 24 26 27 31 33 34 35 36 41 \
                    # --fill_background_with_black

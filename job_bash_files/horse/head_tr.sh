module load python/3.6 cuda cudnn

source /home/aka225/cps_env/bin/activate

python3 -m src.main  --base_dir '/home/aka225/scratch/outputs' \
                    --train \
                    --dataset 'pascal' \
                    --checkpoint_dir '/home/aka225/scratch/outputs/horse_head_te' \
                    --objective_to_optimize 'translator' \
                    --object_name 'horse' \
                    --part_names 'background' 'head' \
                    --train_data_file_ids_file 'self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/horse_train.txt' \
                    --val_data_file_ids_file 'self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/horse_val.txt' \
                    --remove_overlapping_objects \
                    --object_overlapping_threshold 0.05 \
                    --single_object \
                    --optimizer 'Adam' \
                    --epochs 100 \
                    --self_attention_loss_coef 1 \
                    --lr 0.001 \
                    --mask_size 128 \
                    --crop_margin 10 \
                    --crop_threshold 0.2 \
                    --train_data_ids 0 1 2 3 4 5 6 8 10 11 \
                    --val_data_ids 28 29 30 31 33 35 39 40 41 43 \
                    # --fill_background_with_black

module load python/3.6 cuda cudnn

source /home/aka225/cps_env/bin/activate

python3 -m src.main  --base_dir '/home/aka225/scratch/outputs' \
                    --train \
                    --dataset 'pascal' \
                    --checkpoint_dir '/home/aka225/scratch/outputs/car_plate_tr' \
                    --objective_to_optimize 'translator' \
                    --object_name 'car' \
                    --part_names 'background' 'plate' \
                    --train_data_file_ids_file 'self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/car_train.txt' \
                    --val_data_file_ids_file 'self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/car_val.txt' \
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
                    --train_data_ids 3 5 9 11 17 40 41 51 52 54 \
                    --val_data_ids 16 17 18 19 20 22 25 26 27 30 \
                    --fill_background_with_black

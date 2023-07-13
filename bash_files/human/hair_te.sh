module load python/3.6 cuda cudnn

source /home/aka225/cps_env/bin/activate
cd ~/scratch/code/one_shot_segmentation

python3 -m src.main  --base_dir '/home/aka225/scratch/outputs' \
                    --train \
                    --dataset 'celeba-hq' \
                    --checkpoint_dir '/home/aka225/scratch/outputs/human_hair_te' \
                    --objective_to_optimize 'text_embedding' \
                    --part_names 'background' 'hair' \
                    --images_dir '/home/aka225/scratch/data/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img' \
                    --masks_dir '/home/aka225/scratch/data/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno' \
                    --idx_mapping_file '/home/aka225/scratch/data/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt' \
                    --test_file_names_file_path '/home/aka225/scratch/data/CelebAMask-HQ/CelebAMask-HQ/paper_test_file_names.txt' \
                    --train_file_names_file_path '/home/aka225/scratch/data/CelebAMask-HQ/CelebAMask-HQ/non_test_file_names.txt' \
                    --val_file_names_file_path '/home/aka225/scratch/data/CelebAMask-HQ/CelebAMask-HQ/non_test_file_names.txt' \
                    --optimizer 'Adam' \
                    --epochs 40 \
                    --self_attention_loss_coef 1 \
                    --lr 0.1 \
                    --crop_margin 10 \
                    --mask_size 128 \
                    --crop_threshold 0.2 \
                    --val_data_ids 200 201 202 203 204 205 206 207 208 209 \
                    --train_data_ids 1 5 7 8 9 12 13 22 31 34 \

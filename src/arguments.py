import argparse

def add_base_args(parser):
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--attention_layers_to_use', nargs='+', type=str, default=[
                                                                            # 'down_blocks[0].attentions[0].transformer_blocks[0].attn1',
                                                                            # 'down_blocks[0].attentions[0].transformer_blocks[0].attn2',
                                                                            # 'down_blocks[0].attentions[1].transformer_blocks[0].attn1',
                                                                            # 'down_blocks[0].attentions[1].transformer_blocks[0].attn2',
                                                                            # 'down_blocks[1].attentions[0].transformer_blocks[0].attn1',
                                                                            # 'down_blocks[1].attentions[0].transformer_blocks[0].attn2',
                                                                            # 'down_blocks[1].attentions[1].transformer_blocks[0].attn1',
                                                                            # 'down_blocks[1].attentions[1].transformer_blocks[0].attn2',
                                                                            # 'down_blocks[2].attentions[0].transformer_blocks[0].attn1',
                                                                            # 'down_blocks[2].attentions[0].transformer_blocks[0].attn2',  ##########
                                                                            # 'down_blocks[2].attentions[1].transformer_blocks[0].attn1',
                                                                            # 'down_blocks[2].attentions[1].transformer_blocks[0].attn2',  ##########
                                                                            # 'up_blocks[1].attentions[0].transformer_blocks[0].attn1',
                                                                            'up_blocks[1].attentions[0].transformer_blocks[0].attn2',  ########## +
                                                                            # 'up_blocks[1].attentions[1].transformer_blocks[0].attn1',
                                                                            'up_blocks[1].attentions[1].transformer_blocks[0].attn2',  ########## +
                                                                            # 'up_blocks[1].attentions[2].transformer_blocks[0].attn1',
                                                                            'up_blocks[1].attentions[2].transformer_blocks[0].attn2',  ########## +
                                                                            # 'up_blocks[2].attentions[0].transformer_blocks[0].attn1',
                                                                            'up_blocks[2].attentions[0].transformer_blocks[0].attn2',  # +
                                                                            # 'up_blocks[2].attentions[1].transformer_blocks[0].attn1',
                                                                            'up_blocks[2].attentions[1].transformer_blocks[0].attn2',  # +
                                                                            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn1',
                                                                            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn2',
                                                                            'up_blocks[3].attentions[0].transformer_blocks[0].attn1',
                                                                            # 'up_blocks[3].attentions[0].transformer_blocks[0].attn2',
                                                                            'up_blocks[3].attentions[1].transformer_blocks[0].attn1',  #############3
                                                                            # 'up_blocks[3].attentions[1].transformer_blocks[0].attn2',
                                                                            'up_blocks[3].attentions[2].transformer_blocks[0].attn1',
                                                                            # 'up_blocks[3].attentions[2].transformer_blocks[0].attn2',
                                                                            # 'mid_block.attentions[0].transformer_blocks[0].attn1',
                                                                            # 'mid_block.attentions[0].transformer_blocks[0].attn2'
                                                                        ])
        
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--second_gpu_id', type=int, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='pascal', choices=['sample', 'pascal', 'celeba-hq', 'paper_test'])
    parser.add_argument('--noise_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--part_names', nargs='+', type=str, required=True)
    parser.add_argument('--objective_to_optimize', type=str, default='text_embedding')
    parser.add_argument('--log_images', action='store_true', default=False)
    return parser
    
def add_sample_dataset_args(parser):
    parser.add_argument('--src_image_paths', nargs='+', type=str)
    parser.add_argument('--src_mask_paths', nargs='+', type=str)
    parser.add_argument('--target_image_path', nargs='+', type=str)
    return parser

def add_pascal_dataset_args(parser):
    parser.add_argument('--object_name', type=str)
    parser.add_argument('--train_data_file_ids_file', type=str)
    parser.add_argument('--val_data_file_ids_file', type=str)
    parser.add_argument('--blur_background', action='store_true', default=False)
    parser.add_argument('--fill_background_with_black', action='store_true', default=False)
    parser.add_argument('--remove_overlapping_objects', action='store_true', default=False)
    parser.add_argument('--single_object', action='store_true', default=False)
    parser.add_argument('--adjust_bounding_box', action='store_true', default=False)
    parser.add_argument('--final_min_crop_size', type=int)
    parser.add_argument('--object_overlapping_threshold', type=float)
    parser.add_argument('--ann_file_base_dir', type=str)
    parser.add_argument('--images_base_dir', type=str)
    parser.add_argument('--car_test_data_dir', type=str, default='/home/aka225/scratch/data/Car_TestSet')
    return parser

def add_celeba_dataset_args(parser):
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--masks_dir', type=str)
    parser.add_argument('--idx_mapping_file', type=str)
    parser.add_argument('--test_file_names_file_path', type=str)
    parser.add_argument('--train_file_names_file_path', type=str)
    parser.add_argument('--val_file_names_file_path', type=str)
    return parser

def add_paper_test_dataset_args(parser):
    parser.add_argument('--test_images_dir', type=str)
    parser.add_argument('--test_masks_dir', type=str)
    return parser

def add_train_args(parser):
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--self_attention_loss_coef', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--mask_size', type=int, default=128)
    parser.add_argument('--val_data_ids', nargs='+', type=int)
    parser.add_argument('--train_data_ids', nargs='+', type=int)
    return parser

def add_test_args(parser):
    parser.add_argument('--masking', type=str, default='zoomed_masking', choices=['zoomed_masking', 'patched_masking'])
    parser.add_argument('--num_crops_per_side', type=int)
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--crop_margin', default=10, type=int)
    parser.add_argument('--min_square_size', default=200, type=int)
    parser.add_argument('--crop_threshold', type=float)
    parser.add_argument('--zero_pad_test_output', action='store_true', default=False)
    return parser

def init_args():
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    parser = add_sample_dataset_args(parser)
    parser = add_pascal_dataset_args(parser)
    parser = add_celeba_dataset_args(parser)
    parser = add_paper_test_dataset_args(parser)
    parser = add_train_args(parser)
    parser = add_test_args(parser)
    args = parser.parse_args()
    return args
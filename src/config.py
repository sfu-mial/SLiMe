class Config:
    def __init__(self):
        self.src_segmentation_paths = ['/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog6_s1.jpg',
                                       '/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog4_s2.jpg']
        self.src_image_paths = ['/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog6.jpg',
                                '/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog4.jpg']
        self.target_image_path = [f'/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog16.jpg' for i in
                                  range(0, 1)]
        self.base_dir = ""
        self.checkpoint_dir = "/home/aliasgahr/Documents/project/co_part_segmentation/test/ears"
        self.annotations_files_dir = "/home/aliasgahr/Downloads/part_segmentation/trainval/Annotations_Part"

        self.batch_size = 1
        self.lr_1 = 0.4
        self.lr_2 = 0.4
        self.optimizer = "Adam"
        self.epochs = 20
        self.gpu_id = [0]
        self.second_gpu_id = 2
        self.train = False
        self.num_pixels_to_show = 512 * 512
        self.dataset = "sample"  # ["sample", "pascal"]

        self.object_name = "dog"
        self.part_name = "ear"
        self.train_num_crops = 4
        self.test_num_crops = 4
        self.train_data_ids = [4, 5]

        self.mask_size = 256
        self.use_crf = False
        self.fill_background_with_black = False
        self.threshold1 = "mean+std"
        self.threshold2 = "mean+std"
        self.remove_overlapping_objects = False
        self.object_overlapping_threshold = 0.05
        self.train_use_softmax = True
        self.test_use_softmax = True
        self.data_portion = 0.1

        self.attention_layers_to_use = [
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
            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn2',  #+ #-
            # 'up_blocks[3].attentions[0].transformer_blocks[0].attn1',
            # 'up_blocks[3].attentions[0].transformer_blocks[0].attn2', #-
            # 'up_blocks[3].attentions[1].transformer_blocks[0].attn1',
            # 'up_blocks[3].attentions[1].transformer_blocks[0].attn2',
            # 'up_blocks[3].attentions[2].transformer_blocks[0].attn1',
            # 'up_blocks[3].attentions[2].transformer_blocks[0].attn2',
            # 'mid_block.attentions[0].transformer_blocks[0].attn1',
            # 'mid_block.attentions[0].transformer_blocks[0].attn2'
        ]

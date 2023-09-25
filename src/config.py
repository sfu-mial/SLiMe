from glob import glob


class Config:
    def __init__(self):
        # general
        self.base_dir = ""
        self.batch_size = 1
        self.lr_1 = 0.1
        # self.lr_2 = 0.1
        self.lr_t = 0.001
        self.mask_size = 128
        self.attention_layers_to_use = [
            # 'down_blocks[0].attentions[0].transformer_blocks[0].attn1',
            "down_blocks[0].attentions[0].transformer_blocks[0].attn2",
            # 'down_blocks[0].attentions[1].transformer_blocks[0].attn1',
            "down_blocks[0].attentions[1].transformer_blocks[0].attn2",
            # 'down_blocks[1].attentions[0].transformer_blocks[0].attn1',
            "down_blocks[1].attentions[0].transformer_blocks[0].attn2",
            # 'down_blocks[1].attentions[1].transformer_blocks[0].attn1',
            "down_blocks[1].attentions[1].transformer_blocks[0].attn2",
            # 'down_blocks[2].attentions[0].transformer_blocks[0].attn1',
            "down_blocks[2].attentions[0].transformer_blocks[0].attn2",
            # 'down_blocks[2].attentions[1].transformer_blocks[0].attn1',
            "down_blocks[2].attentions[1].transformer_blocks[0].attn2",
            # 'up_blocks[1].attentions[0].transformer_blocks[0].attn1',
            # "up_blocks[1].attentions[0].transformer_blocks[0].attn2",  ##########
            # 'up_blocks[1].attentions[1].transformer_blocks[0].attn1',
            # "up_blocks[1].attentions[1].transformer_blocks[0].attn2",  ##########
            # 'up_blocks[1].attentions[2].transformer_blocks[0].attn1',
            # "up_blocks[1].attentions[2].transformer_blocks[0].attn2",  ##########
            # 'up_blocks[2].attentions[0].transformer_blocks[0].attn1',
            # "up_blocks[2].attentions[0].transformer_blocks[0].attn2",  ##########
            # 'up_blocks[2].attentions[1].transformer_blocks[0].attn1',
            # "up_blocks[2].attentions[1].transformer_blocks[0].attn2",  ##########
            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn1',
            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn2',
            "up_blocks[3].attentions[0].transformer_blocks[0].attn1",
            # 'up_blocks[3].attentions[0].transformer_blocks[0].attn2',
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",  #############3
            # 'up_blocks[3].attentions[1].transformer_blocks[0].attn2',
            "up_blocks[3].attentions[2].transformer_blocks[0].attn1",
            # 'up_blocks[3].attentions[2].transformer_blocks[0].attn2',
            # 'mid_block.attentions[0].transformer_blocks[0].attn1',
            # 'mid_block.attentions[0].transformer_blocks[0].attn2'
        ]
        # self.lr_conv = 0.0005
        self.optimizer = "Adam"
        self.epochs = 40
        self.gpu_id = [0]
        self.second_gpu_id = 1
        self.train = True
        self.dataset = "celeba-hq"  # ["sample", "pascal", "celeba-hq", "paper_test"]
        self.num_translator_hidden_layers = 1
        self.translator_hidden_layer_dim = 1024
        self.use_dropout_in_hidden_layer = False
        # self.noise_path = "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/stable_diffusion_2/noise1.pth"
        self.noise_path = None

        # Only for sample dataset
        # self.src_image_paths = sorted(glob("/home/aliasgahr/Downloads/co_part_segmentation_data/horse/*.jpg"))[-10:]
        self.src_image_paths = [
            f"/home/aliasgahr/Downloads/co_part_segmentation_data/horse/horse_{i}.jpg"
            for i in range(1, 11)
        ]
        # self.src_mask_paths = sorted(glob("/home/aliasgahr/Downloads/co_part_segmentation_data/horse/horse_tail_*.png"))[-10:]
        self.src_mask_paths = [
            "/home/aliasgahr/Downloads/co_part_segmentation_data/car/train_data/car_whole_17.png"
        ]
        self.target_image_path = [
            f"/home/aliasgahr/Downloads/co_part_segmentation_data/car/train_data/car_{i}.jpg"
            for i in range(1, 27)
        ]

        # For training
        self.train_checkpoint_dir = "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/stable_diffusion_2/human_brow_2"
        # self.train_part_names = ["background", "mouth", "nose", "brow", "ear"]
        self.train_part_names = ["background", "brow"]
        # self.train_part_names = ["background", "eye", "mouth", "nose", "brow", "ear", "skin", "neck", "cloth", "hair"]
        # self.train_part_names = ["background", "body", "light", "plate", "wheel", "window"]
        # self.train_part_names = ["background", "head", "leg", "neck+torso", "tail"]
        # self.train_part_names = ['background', 'head', 'nose', 'torso', 'tail', 'neck', 'leg', 'paw', 'ear', 'eye', 'muzzl']
        # self.train_part_names = ['background', 'head', 'nose', 'ear', 'eye']
        self.self_attention_loss_coef = 1

        # For testing
        #     self.test_checkpoint_dir = ["/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/dog_nose"]
        #     self.test_part_names = ["background", "nose"]
        #     self.test_checkpoint_dir = ["/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/car_whole_t_hl_bg",
        # "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/car_light1",
        # "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/car_plate1",
        # "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/car_wheel1",
        # "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/car_window1",
        # ]
        # self.test_part_names = ["background", "body", "light", "plate", "wheel", "window"]
        # self.test_checkpoint_dir = ["/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/horse_whole_t_hl_one",
        #                         "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/horse_leg",
        #                         "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/horse_neck_torso",
        #                         "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/horse_tail",
        #                         ]
        self.test_checkpoint_dir = "/home/aliasgahr/Documents/project/co_part_segmentation/checkpoints/stable_diffusion_2/human_brow_2"
        # self.test_part_names = ["background", "eye", "mouth", "nose", "brow", "ear", "skin", "neck", "cloth", "hair"]
        # self.test_part_names = ["background", "body", "light", "plate", "wheel", "window"]
        # self.test_part_names = ["background", "head", "leg", "neck+torso", "tail"]
        # self.test_part_names = ["background", "mouth", "nose", "brow", "ear"]
        self.test_part_names = ["background", "brow"]
        # self.test_part_names = ['background', 'head', 'nose', 'torso', 'tail', 'neck', 'leg', 'paw', 'ear', 'eye', 'muzzl']
        # self.test_part_names = ['background', 'head', 'nose', 'ear', 'eye']
        self.num_patchs_per_side = 2
        self.patch_size = 300
        self.patch_threshold = 0.2
        self.masking = "zoomed_masking"

        # Only for pascal dataset
        self.object_name = "horse"
        self.train_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/horse_train.txt"
        self.val_data_file_ids_file = "/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/horse_val.txt"
        self.blur_background = False
        self.fill_background_with_black = False
        self.remove_overlapping_objects = True
        self.object_overlapping_threshold = 0.05
        self.min_crop_size = 400
        self.single_object = True
        self.adjust_bounding_box = False

        # self.train_part_names = ["background", "head", "leg", "neck+torso", "tail"]
        # first set of runs use mean+2std
        # self.train_data_ids = [3, 4, 10, 15, 18, 23, 24, 30, 31, 32]  # wheels (38.41 | 473) ---- (26.37, 48.95, 57.05 | 529) +
        # self.train_data_ids = [3, 5, 9, 16, 17, 21, 23, 51, 52, 54]  # lights (52.6 | 486) ---- (12.45, 30.50, 39.25 | 528) +
        # self.train_data_ids = [5, 8, 11, 15, 17, 54, 101, 103, 108, 202]  # window (50.02 | 493)  ---- (42.04, 59.35, 57.74 | 530) -
        # self.train_data_ids = [3, 4, 7, 9, 11, 201, 202, 203, 206, 209]  # plate (11.24 | 488) ---- (3.97, 7.92, 11.59 | 527) -
        # self.train_data_ids = [3, 4, 5, 8, 9, 10, 201, 202, 206, 214]  # head (42.63 | 492) --- (31.65, 49.47, 53.33 | 533) +
        # self.train_data_ids = [3, 5, 9, 13, 14, 15, 477, 481, 483, 484]  # neck+torso (25.56 | 521) ---- (55.76, 53.22, 38.08 | 532) -
        # self.train_data_ids = [4, 5, 8, 9, 11, 12, 301, 302, 303, 407]  # legs 38.26 | 522 ---- (21.06, 38.38, 42.81 | 531) -
        # self.train_data_ids = [1, 2, 3, 7, 9, 207, 208, 209, 213, 220]  # tail (31.23 | 523) ----  (13.74 , 28.25 , 33.07 | 524) +
        # self.train_data_ids = [3, 7, 10, 11, 12, 201, 202, 209, 210, 212]  # plate2 ( 8.93, 26.99, 36.59 | 537) +
        # self.train_data_ids = [5, 9, 205, 210, 222, 224, 227, 228, 233, 302]  # body ( | 536)

        # self.train_data_ids = [1, 7, 8, 10, 18, 21, 24, 29, 30, 34]  # car body
        # self.val_data_ids = [43, 45, 46, 47, 48, 49, 51, 54, 55, 56]  # car body
        # self.train_data_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]  # car light
        # self.val_data_ids = [11, 14, 16, 17, 18, 19, 20, 21, 24, 25]  # car light
        # self.train_data_ids = [3, 5, 9, 11, 17, 40, 41, 51, 52, 54]  # car plate
        # self.val_data_ids = [16, 17, 18, 19, 20, 22, 25, 26, 27, 30]  # car plate
        # self.train_data_ids = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]  # car wheel
        # self.val_data_ids = [12, 13, 14, 17, 19, 20, 21, 22, 29, 31]  # car wheel
        # self.train_data_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # car window
        # self.val_data_ids = [10, 11, 13, 15, 16, 17, 19, 28, 29, 32]  # car window

        # self.train_data_ids = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # nose
        # self.train_data_ids = [71]  # whole car one
        # self.train_data_ids = [7, 8, 12, 19, 30, 35, 52, 56, 58, 69]  # whole car
        # self.train_data_ids = [2]  # whole horse one
        # self.train_data_ids = [2, 3, 4, 5, 6, 12, 18, 20, 25, 27]  # whole horse
        # self.val_data_ids = [48, 55, 71, 73, 74, 89, 94, 99, 100, 103]  # whole car
        # self.val_data_ids = [28, 29, 30, 31, 33, 35, 39, 40, 41, 43]  # whole horse
        # self.train_data_ids = [6]  # dog head one
        # self.train_data_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 22]  # dog head
        # self.val_data_ids = [i for i in range(11, 21)]  # dog head

        # self.train_data_ids = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11]  # horse head
        # self.val_data_ids = [28, 29, 30, 31, 33, 35, 39, 40, 41, 43]  # horse head
        # self.train_data_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]  # horse leg
        # self.val_data_ids = [11, 12, 13, 15, 17, 18, 19, 20, 21, 23]  # horse leg
        # self.train_data_ids = [1, 2, 3, 4, 5, 6, 8, 11, 13, 16]  # horse neck_torso
        # self.val_data_ids = [18, 19, 20, 22, 23, 24, 26, 27, 29, 30]  # horse neck_torso
        # self.train_data_ids = [1, 3, 4, 5, 13, 14, 16, 17, 18, 22]  # horse tail
        # self.val_data_ids = [23, 24, 26, 27, 31, 33, 34, 35, 36, 41]  # horse tail

        # only for celeba dataset
        self.images_dir = "/home/aliasgahr/Downloads/CelebAMask-HQ/CelebA-HQ-img"
        self.masks_dir = (
            "/home/aliasgahr/Downloads/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
        )
        self.idx_mapping_file = (
            "/home/aliasgahr/Downloads/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"
        )
        self.test_file_names_file_path = (
            "/home/aliasgahr/Downloads/CelebAMask-HQ/paper_test_file_names.txt"
        )
        self.train_file_names_file_path = (
            "/home/aliasgahr/Downloads/CelebAMask-HQ/non_test_file_names.txt"
        )
        self.val_file_names_file_path = (
            "/home/aliasgahr/Downloads/CelebAMask-HQ/non_test_file_names.txt"
        )
        # self.train_data_ids = [i for i in range(10)]
        self.val_data_ids = [i for i in range(200, 210)]
        # self.train_data_ids = [0, 1, 8, 9, 12, 36, 50, 57, 67, 70]  # eye
        # self.train_data_ids = [0, 1, 5, 9, 12, 36, 60, 75, 83, 84]  # mouth
        # self.train_data_ids = [0, 1, 5, 10, 8, 9, 12, 36, 75, 83]  # nose
        self.train_data_ids = [0, 1, 8, 9, 12, 36, 57, 75, 82, 83]  # brow
        # self.train_data_ids = [5, 7, 8, 9, 12, 32, 36, 48, 57, 67]  # ear
        # self.train_data_ids = [0, 1, 7, 8, 9, 12, 36, 59, 75, 85]  # skin
        # self.train_data_ids = [0, 1, 7, 8, 9, 36, 75, 81, 85, 87]  # neck
        # self.train_data_ids = [7, 8, 9, 12, 14, 29, 36, 57, 86, 87]  # cloth
        # self.train_data_ids = [1, 5, 7, 8, 9, 12, 13, 22, 31, 34]  # hair

        # train on my segmented data
        # 538 horse head         -> (31.50, 50.31, 52.95 | 542) +
        # 539 horse legs         -> (21.96, 39.85, 44.14 | 543) -
        # 540 horse tail         -> (12.58, 26.65, 33.03 | 544) +
        # 541 horse neck+torso   -> (52.41, 49.11, 35.29 | 545) -

        # 546 car light          -> (11.63, 29.77, 38.14 | 551) +
        # 547 car plate          -> (8.35, 24.26, 33.00 | 552) +
        # 548 car wheel          -> (26.83, 49.02, 52.18 | 553) -
        # 549 car window         -> (20.17, 28.23, 28.52) -
        # 550 car body           -> ()

        # train on my segmented data with crop augmentation
        # 604 horse head         -> (56.2 | 608) +
        # 602 horse legs         -> (38.41 | 608) -
        # 606 horse tail         -> (28.67 | 608) +
        # 605 horse neck+torso   -> (63.52 | 608) -

        # 600 car light          -> (32.12 | 607) +
        # 603 car plate          -> (32.49 | 607) +
        # 599 car wheel          -> (48.86 | 607) -
        # 595 car window         -> (55.45 | 607) -
        # 596 car body           -> (57.62 | 607) -

        # Paper Test Dataset
        self.test_images_dir = "/home/aliasgahr/Downloads/Car_TestSet/image_bg"
        self.test_masks_dir = "/home/aliasgahr/Downloads/Car_TestSet/gt_mask"

class Config:
    def __init__(self):
        self.point_location2 = [418, 185]  # (y, x)
        self.point_location1 = [418, 185]  # (y, x)
        self.src_segmentation_path = '/home/aliasgahr/Downloads/co_part_segmentation_data/car/car_source_w.png'
        self.src_image_path = '/home/aliasgahr/Downloads/co_part_segmentation_data/car/car_source.png'
        self.target_image_path = '/home/aliasgahr/Downloads/co_part_segmentation_data/car/car5.jpg'
        self.base_dir = ""
        self.checkpoint_dir = ""
        self.annotations_files_dir = "/home/aliasgahr/Downloads/part_segmentation/trainval/Annotations_Part"

        self.lr = 0.1
        self.optimizer = "Adam"
        self.epochs = 40
        self.batch_size = 1
        self.gpu_id = [0]
        self.second_gpu_id = 1
        self.train = False
        self.num_pixels_to_show = 512*512
        self.dataset = "pascal"  # ["sample", "pascal"]

        self.object_name = "car"
        self.part_name = "window"
        self.train_num_crops = 10
        self.test_num_crops = 1
        self.train_crop_ratio = 0.93
        self.test_crop_ratio = 1
        self.train_data_id = 2

        self.mask_size = 256
        self.use_crf = False

class Config:
    def __init__(self):
        self.point_location2 = [418, 185]  # (y, x)
        self.point_location1 = [418, 185]  # (y, x)
        self.src_segmentation_path = '/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog4_s4.jpg'
        self.src_image_path = '/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog4.jpg'
        self.target_image_path = '/home/aliasgahr/Downloads/co_part_segmentation_data/dogs/dog5.jpg'
        self.base_dir = ""
        self.checkpoint_dir = ""

        self.lr = 0.1
        self.optimizer = "Adam"
        self.epochs = 40
        self.batch_size = 1
        self.num_crops = 5
        self.gpu_id = [0]
        self.train = True
        self.crop_ratio = 0.93
        self.num_pixels_to_show = 512*512

class Config:
    def __init__(self):
        self.point_location = [int((300/786)*512), int((499/786)*512)]  # (y, x)
        self.src_image_path = "/home/aliasgahr/Downloads/cat_1.jpg"
        self.target_image_path = "/home/aliasgahr/Downloads/cat.jpg"
        self.base_dir = "./"

        self.lr = 2.37*1e-3
        self.optimizer = "Adam"
        self.epochs = 10
        self.batch_size = 1
        self.num_crops = 4
        self.gpu_id = [2]
        self.train = False
        self.crop_ratio = 0.93

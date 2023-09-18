import torch.nn as nn


class PixelClassifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(PixelClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # nn.BatchNorm1d(num_features=128),
            nn.Conv2d(32, numpy_class, 3, 1, 1),
            # nn.ReLU(),
            # nn.Dropout(p=0.4),
            # nn.BatchNorm1d(num_features=32),
            # nn.Linear(32, numpy_class),
        )

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)


# class PixelClassifier(nn.Module):
#     def __init__(self, numpy_class, dim):
#         super(PixelClassifier, self).__init__()
#         if numpy_class < 30:
#             self.layers = nn.Sequential(
#                 nn.Linear(dim, 256),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.2),
#                 # nn.BatchNorm1d(num_features=128),
#                 nn.Linear(256, 32),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.4),
#                 # nn.BatchNorm1d(num_features=32),
#                 nn.Linear(32, numpy_class),
#             )
#         else:
#             self.layers = nn.Sequential(
#                 nn.Linear(dim, 256),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(num_features=256),
#                 nn.Linear(256, 128),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(num_features=128),
#                 nn.Linear(128, numpy_class),
#             )

#     def init_weights(self, init_type="normal", gain=0.02):
#         """
#         initialize network's weights
#         init_type: normal | xavier | kaiming | orthogonal
#         https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
#         """

#         def init_func(m):
#             classname = m.__class__.__name__
#             if hasattr(m, "weight") and (
#                 classname.find("Conv") != -1 or classname.find("Linear") != -1
#             ):
#                 if init_type == "normal":
#                     nn.init.normal_(m.weight.data, 0.0, gain)
#                 elif init_type == "xavier":
#                     nn.init.xavier_normal_(m.weight.data, gain=gain)
#                 elif init_type == "kaiming":
#                     nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#                 elif init_type == "orthogonal":
#                     nn.init.orthogonal_(m.weight.data, gain=gain)

#                 if hasattr(m, "bias") and m.bias is not None:
#                     nn.init.constant_(m.bias.data, 0.0)

#             elif classname.find("BatchNorm2d") != -1:
#                 nn.init.normal_(m.weight.data, 1.0, gain)
#                 nn.init.constant_(m.bias.data, 0.0)

#         self.apply(init_func)

#     def forward(self, x):
#         return self.layers(x)

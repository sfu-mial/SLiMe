import os

import pytorch_lightning as pl
import torch
import torchvision
from torch import optim

from src.config import Config
from src.crf import crf
from src.stable_difusion import StableDiffusion
from src.utils import get_square_cropping_coords, calculate_iou, post_process_attention_map


class CoSegmenterTrainer(pl.LightningModule):
    def __init__(self, config: Config, learning_rate=0.001):
        super().__init__()
        self.counter = 0
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.automatic_optimization = True
        self.stable_diffusion = StableDiffusion(sd_version='1.4', partial_run=False,
                                                attention_layers_to_use=config.attention_layers_to_use)

        self.target_image_original = None

        self.min_loss_1 = 10000000
        self.min_loss_2 = 10000000
        self.generate_noise = True
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.first_binarized_attention_map = None

        if self.config.train:
            uncond_embeddings, text_embeddings = self.stable_diffusion.get_text_embeds("object part", "object part")
            self.token_t = text_embeddings[:, 2:3].clone()
            self.token_t.requires_grad_(True)
            self.text_embedding = torch.cat([text_embeddings[:, :2], self.token_t, text_embeddings[:, 3:]], dim=1)

            self.token_u = uncond_embeddings[:, 2:3].clone()
            self.token_u.requires_grad_(True)
            self.uncond_embedding = torch.cat([uncond_embeddings[:, :2], self.token_u, uncond_embeddings[:, 3:]], dim=1)

    def on_train_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))

    def training_step(self, batch, batch_idx):
        src_images, mask1, mask2 = batch
        print(mask1.unique(), mask2.unique())
        small_mask2 = torch.nn.functional.interpolate(mask2[None, ...], 64, mode="nearest")[0]
        mask1 = mask1[0]
        mask2 = mask2[0]
        small_mask2 = small_mask2[0]
        self.text_embedding = torch.cat([self.text_embedding[:, :2], self.token_t, self.text_embedding[:, 3:]], dim=1)
        self.uncond_embedding = torch.cat([self.uncond_embedding[:, :2], self.token_u, self.uncond_embedding[:, 3:]],
                                          dim=1)

        loss, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps = self.stable_diffusion.train_step(
            torch.repeat_interleave(torch.cat([self.uncond_embedding, self.text_embedding]), self.config.batch_size, 0),
            src_images, t=torch.tensor(260),
            back_propagate_loss=False, generate_new_noise=self.generate_noise,
            attention_output_size=self.config.mask_size, token_id=2)
        self.generate_noise = False

        loss3 = 0
        if sd_self_attention_maps is not None:
            self_attention_map = sd_self_attention_maps[torch.where(small_mask2.flatten() == 0.1)[0]].mean(dim=0)
            loss3 = torch.nn.functional.mse_loss(self_attention_map, small_mask2)

        loss1 = torch.nn.functional.mse_loss(sd_cross_attention_maps1, mask1)
        loss2 = torch.nn.functional.mse_loss(sd_cross_attention_maps2, mask2)

        if loss1 < self.min_loss_1:
            self.min_loss_1 = loss1
            torch.save(self.text_embedding, os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_1.pth"))
        if loss2 < self.min_loss_2:
            self.min_loss_2 = loss2
            torch.save(self.uncond_embedding,
                       os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_2.pth"))

        loss = loss1 + loss2 + loss3

        self.log("mse_loss1", loss1, on_step=True, sync_dist=True)
        self.log("mse_loss2", loss2, on_step=True, sync_dist=True)
        self.log("mse_loss3", loss3, on_step=True, sync_dist=True)

        sd_cross_attention_maps1 = (sd_cross_attention_maps1 - sd_cross_attention_maps1.min()) / (
                sd_cross_attention_maps1.max() - sd_cross_attention_maps1.min())
        sd_cross_attention_maps2 = (sd_cross_attention_maps2 - sd_cross_attention_maps2.min()) / (
                sd_cross_attention_maps2.max() - sd_cross_attention_maps2.min())
        if sd_self_attention_maps is not None:
            self_attention_map = (self_attention_map - self_attention_map.min()) / (self_attention_map.max() - self_attention_map.min())
            self_attention_map_grid = torchvision.utils.make_grid(self_attention_map[None, ...])
            self.logger.experiment.add_image("train self attention map", self_attention_map_grid, self.counter)

        images_grid = torchvision.utils.make_grid(src_images)
        sd_attention_maps_grid1 = torchvision.utils.make_grid(sd_cross_attention_maps1[None, ...])
        sd_attention_maps_grid2 = torchvision.utils.make_grid(sd_cross_attention_maps2[None, ...])
        mask1_grid = torchvision.utils.make_grid(mask1[None, ...])
        mask2_grid = torchvision.utils.make_grid(mask2[None, ...])
        mask3_grid = torchvision.utils.make_grid(small_mask2[None, ...])

        self.logger.experiment.add_image("train image", images_grid, 0)
        self.logger.experiment.add_image("train sd attention maps1", sd_attention_maps_grid1, self.counter)
        self.logger.experiment.add_image("train sd attention maps2", sd_attention_maps_grid2, self.counter)
        self.logger.experiment.add_image("train mask1", mask1_grid, self.counter)
        self.logger.experiment.add_image("train mask2", mask2_grid, self.counter)
        self.logger.experiment.add_image("train mask3", mask3_grid, self.counter)
        self.counter += 1

        return loss

    def on_train_end(self) -> None:
        torch.save(self.stable_diffusion.noise, os.path.join(self.config.checkpoint_dir, "noise.pth"))

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))
        self.stable_diffusion.change_hooks(attention_layers_to_use=self.config.attention_layers_to_use[:-1])  # exclude self attention layer
        self.text_embedding = torch.load(os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_1.pth")).to(
            self.device)
        self.uncond_embedding = torch.load(
            os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_2.pth")).to(
            self.device)
        noise = torch.load(os.path.join(self.config.checkpoint_dir, "noise.pth"))
        self.stable_diffusion.noise = noise

    def get_attention_maps(self, image, y_start, y_end, x_start, x_end):
        with torch.no_grad():
            loss, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps = self.stable_diffusion.train_step(
                torch.cat([self.uncond_embedding, self.text_embedding], dim=0), image,
                t=torch.tensor(20), back_propagate_loss=False, generate_new_noise=False, attention_output_size=512,
                token_id=2)
        original_size_attention_map1 = post_process_attention_map(
            sd_cross_attention_maps1,
            [y_start, y_end, x_start, x_end],
        )
        original_size_attention_map2 = post_process_attention_map(
            sd_cross_attention_maps2,
            [y_start, y_end, x_start, x_end],
        )

        if self.config.threshold1 == "mean+std":
            threshold = torch.mean(original_size_attention_map1) + 1 * torch.std(original_size_attention_map1)
        elif isinstance(self.config.threshold1, float):
            threshold = self.config.threshold1
        binarized_attention_map1 = torch.where(original_size_attention_map1 > threshold, 1, 0)

        if self.config.threshold2 == "mean+std":
            threshold = torch.mean(original_size_attention_map2) + 1 * torch.std(original_size_attention_map2)
        elif isinstance(self.config.threshold2, float):
            threshold = self.config.threshold2
        binarized_attention_map2 = torch.where(original_size_attention_map2 > threshold, 1, 0)

        return original_size_attention_map1, original_size_attention_map2, binarized_attention_map1, binarized_attention_map2

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask_provided = not torch.all(mask == 0)
        if mask_provided:
            mask = mask[0]

        original_size_attention_map1, original_size_attention_map2, binarized_attention_map1, binarized_attention_map2 = self.get_attention_maps(image, 0, 512, 0, 512)

        attention_grid1 = torchvision.utils.make_grid(original_size_attention_map1)
        attention_grid2 = torchvision.utils.make_grid(original_size_attention_map2)

        masked_image_grid1 = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - binarized_attention_map1[None, ...]).cpu() + torch.stack(
                [binarized_attention_map1 * 0, binarized_attention_map1 * 0,
                 binarized_attention_map1], dim=0))
        masked_image_grid2 = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - binarized_attention_map2[None, ...]).cpu() + torch.stack(
                [binarized_attention_map2 * 0, binarized_attention_map2 * 0,
                 binarized_attention_map2], dim=0))

        image_grid = torchvision.utils.make_grid(image)

        log_id = self.config.test_num_crops * batch_idx

        self.logger.experiment.add_image("test attention map1", attention_grid1, log_id)
        self.logger.experiment.add_image("test attention map2", attention_grid2, log_id)
        self.logger.experiment.add_image("test masked image1", masked_image_grid1, log_id)
        self.logger.experiment.add_image("test masked image2", masked_image_grid2, log_id)
        self.logger.experiment.add_image("test image", image_grid, log_id)
        coords = []
        if self.config.test_num_crops > 1:
            x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(binarized_attention_map2)
            coords = [[y_start, y_end, x_start, x_end]]
            for square_size in torch.linspace(crop_size, 512, self.config.test_num_crops)[1:-1]:
                x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(binarized_attention_map2,
                                                                                            square_size=square_size)
                coords.append([y_start, y_end, x_start, x_end])

        sd_multi_scale_cross_attention_maps1 = original_size_attention_map1
        sd_multi_scale_cross_attention_maps2 = original_size_attention_map2
        for idx, (y_start, y_end, x_start, x_end) in enumerate(coords[::-1]):
            cropped_image = image[:, :, y_start:y_end, x_start:x_end]
            cropped_image = torch.nn.functional.interpolate(cropped_image, 512, mode="bilinear")
            original_size_attention_map1, original_size_attention_map2, binarized_attention_map1, binarized_attention_map2 = self.get_attention_maps(
                cropped_image, y_start, y_end, x_start, x_end)

            sd_multi_scale_cross_attention_maps1 += ((idx+2) * original_size_attention_map1)
            sd_multi_scale_cross_attention_maps2 += ((idx+2) * original_size_attention_map2)

            attention_grid1 = torchvision.utils.make_grid(original_size_attention_map1)
            attention_grid2 = torchvision.utils.make_grid(original_size_attention_map2)

            masked_image_grid1 = torchvision.utils.make_grid(
                image[0].cpu() * (
                            1 - binarized_attention_map1[None, ...]).cpu() + torch.stack(
                    [binarized_attention_map1 * 0, binarized_attention_map1 * 0,
                     binarized_attention_map1], dim=0))
            masked_image_grid2 = torchvision.utils.make_grid(
                image[0].cpu() * (
                            1 - binarized_attention_map2[None, ...]).cpu() + torch.stack(
                    [binarized_attention_map2 * 0, binarized_attention_map2 * 0,
                     binarized_attention_map2], dim=0))

            cropped_image_grid = torchvision.utils.make_grid(cropped_image)

            log_id += 1

            self.logger.experiment.add_image("test attention map1", attention_grid1, log_id)
            self.logger.experiment.add_image("test attention map2", attention_grid2, log_id)
            self.logger.experiment.add_image("test masked image1", masked_image_grid1, log_id)
            self.logger.experiment.add_image("test masked image2", masked_image_grid2, log_id)
            self.logger.experiment.add_image("test image", cropped_image_grid, log_id)

        sd_multi_scale_cross_attention_maps1 /= sum(range(self.config.test_num_crops+1))
        sd_multi_scale_cross_attention_maps2 /= sum(range(self.config.test_num_crops+1))

        if self.config.threshold1 == "mean+std":
            threshold = sd_multi_scale_cross_attention_maps1.mean() + 2 * sd_multi_scale_cross_attention_maps1.std()
        elif isinstance(self.config.threshold1, float):
            threshold = self.config.threshold1
        binarized_attention_map1 = torch.where(
            sd_multi_scale_cross_attention_maps1 > threshold, 1., 0.)

        if self.config.threshold2 == "mean+std":
            threshold = sd_multi_scale_cross_attention_maps2.mean() + 2 * sd_multi_scale_cross_attention_maps2.std()
        elif isinstance(self.config.threshold2, float):
            threshold = self.config.threshold2
        binarized_attention_map2 = torch.where(
            sd_multi_scale_cross_attention_maps2 > threshold, 1., 0.)

        if mask_provided:
            if self.config.use_crf:
                crf_mask = torch.as_tensor(
                    crf((image[0].permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy().copy(
                        order='C'),
                        torch.stack([binarized_attention_map1, binarized_attention_map1,
                                     binarized_attention_map1],
                                    dim=2).type(torch.float).numpy()))[:, :, 0]
                crf_masked_image_grid1 = torchvision.utils.make_grid(
                    image[0].cpu() * (1 - crf_mask[None, ...]).cpu() + crf_mask[None, ...])
                iou1 = calculate_iou((torch.nn.functional.interpolate(crf_mask[None, None, ...],
                                                                     self.config.mask_size)[0, 0] > 0).type(
                    torch.uint8), (mask.cpu() > 0).type(torch.uint8))
            else:
                iou1 = calculate_iou((torch.nn.functional.interpolate(
                    binarized_attention_map1[None, None, ...], self.config.mask_size)[0, 0] > 0).type(
                    torch.uint8), (mask.cpu() > 0).type(torch.uint8))
                iou2 = calculate_iou(
                    (torch.nn.functional.interpolate(binarized_attention_map2[None, None, ...],
                                                     self.config.mask_size)[0, 0] > 0).type(
                        torch.uint8), (mask.cpu() > 0).type(torch.uint8))
            masks_grid = torchvision.utils.make_grid(mask[None, ...])

        masked_image_grid1 = torchvision.utils.make_grid(
            image[0].cpu() * (1 - binarized_attention_map1[None, ...]).cpu() + torch.stack(
                [binarized_attention_map1 * 0, binarized_attention_map1 * 0, binarized_attention_map1], dim=0))
        masked_image_grid2 = torchvision.utils.make_grid(
            image[0].cpu() * (1 - binarized_attention_map2[None, ...]).cpu() + torch.stack(
                [binarized_attention_map2 * 0, binarized_attention_map2 * 0, binarized_attention_map2], dim=0))

        attention_grid1 = torchvision.utils.make_grid(sd_multi_scale_cross_attention_maps1)
        attention_grid2 = torchvision.utils.make_grid(sd_multi_scale_cross_attention_maps2)

        if mask_provided:
            self.logger.experiment.add_image("test mask", masks_grid, batch_idx)
            self.log("test iou1", iou1, on_step=True, sync_dist=True)
            self.log("test iou2", iou2, on_step=True, sync_dist=True)
            if self.config.use_crf:
                self.logger.experiment.add_image("test crf masked image1", crf_masked_image_grid1, batch_idx)

        self.logger.experiment.add_image("test final masked image1", masked_image_grid1, batch_idx)
        self.logger.experiment.add_image("test final masked image2", masked_image_grid2, batch_idx)

        self.logger.experiment.add_image("test final attention maps1", attention_grid1, batch_idx)
        self.logger.experiment.add_image("test final attention maps2", attention_grid2, batch_idx)

        return torch.tensor(0.)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            [
                {'params': self.token_t, 'lr': self.config.lr_1},
                {'params': self.token_u, 'lr': self.config.lr_2},

            ],
            lr=self.config.lr_1,
        )
        return {"optimizer": optimizer}

import os

import pytorch_lightning as pl
import torch
import torchvision
from torch import optim

from src.config import Config
from src.crf import crf
from src.stable_difusion import StableDiffusion
from src.utils import calculate_iou, post_process_attention_map, get_crops_coords


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
        small_mask2 = torch.nn.functional.interpolate(mask2[None, ...], 64, mode="nearest")[0]
        mask1 = mask1[0]
        mask2 = mask2[0]
        small_mask2 = small_mask2[0]
        self.text_embedding = torch.cat([self.text_embedding[:, :2], self.token_t, self.text_embedding[:, 3:]], dim=1)
        self.uncond_embedding = torch.cat([self.uncond_embedding[:, :2], self.token_u, self.uncond_embedding[:, 3:]],
                                          dim=1)

        loss, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps = self.stable_diffusion.train_step(
            torch.repeat_interleave(torch.cat([self.uncond_embedding, self.text_embedding]), self.config.batch_size, 0),
            src_images, t=torch.tensor(20),
            back_propagate_loss=False, generate_new_noise=self.generate_noise,
            attention_output_size=self.config.mask_size, token_id=2)
        self.generate_noise = False

        loss3 = 0
        if sd_self_attention_maps is not None:
            self_attention_map = sd_self_attention_maps[0][torch.where(small_mask2.flatten() == 0.1)[0]].mean(dim=0)
            loss3 = torch.nn.functional.mse_loss(self_attention_map, small_mask2)

        sd_cross_attention_maps1 = (sd_cross_attention_maps1 - sd_cross_attention_maps1.min()) / (
                sd_cross_attention_maps1.max() - sd_cross_attention_maps1.min())

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
        self.stable_diffusion.change_hooks(attention_layers_to_use=self.config.attention_layers_to_use)  # exclude self attention layer
        self.text_embedding = torch.load(os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_1.pth")).to(
            self.device)
        self.uncond_embedding = torch.load(
            os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_2.pth")).to(
            self.device)
        noise = torch.load(os.path.join(self.config.checkpoint_dir, "noise.pth")).to(self.device)
        self.stable_diffusion.noise = noise

    def get_attention_maps(self, image, y_start, y_end, x_start, x_end, threshold1, threshold2):
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

        if threshold1 == "mean+std":
            threshold = torch.mean(original_size_attention_map1) + 1 * torch.std(original_size_attention_map1)
        if threshold1 == "mean+2std":
            threshold = torch.mean(original_size_attention_map1) + 2 * torch.std(original_size_attention_map1)
        elif isinstance(self.config.threshold1, float):
            threshold = self.config.threshold1
        binarized_attention_map1 = torch.where(original_size_attention_map1 > threshold, 1, 0)

        if threshold2 == "mean+std":
            threshold = torch.mean(original_size_attention_map2) + 1 * torch.std(original_size_attention_map2)
        if threshold2 == "mean+2std":
            threshold = torch.mean(original_size_attention_map2) + 2 * torch.std(original_size_attention_map2)
        elif isinstance(self.config.threshold2, float):
            threshold = self.config.threshold2
        binarized_attention_map2 = torch.where(original_size_attention_map2 > threshold, 1, 0)

        return original_size_attention_map1, original_size_attention_map2, binarized_attention_map1, binarized_attention_map2

    def get_patched_masks(self, image, crop_size, num_crops_per_side, threshold):
        crops_coords = get_crops_coords(image.shape[2:], crop_size,
                                        num_crops_per_side)

        max_attention_value = 0
        final_attention_map = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)
        aux_attention_map = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)
        for i in range(num_crops_per_side):
            for j in range(num_crops_per_side):
                y_start, y_end, x_start, x_end = crops_coords[i * num_crops_per_side + j]
                cropped_image = image[:, :, y_start:y_end, x_start:x_end]
                cropped_image = torch.nn.functional.interpolate(cropped_image, 512, mode="bilinear")
                with torch.no_grad():
                    loss, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps = self.stable_diffusion.train_step(
                        torch.cat([self.uncond_embedding, self.text_embedding], dim=0), cropped_image,
                        t=torch.tensor(10), back_propagate_loss=False, generate_new_noise=False,
                        attention_output_size=64,
                        token_id=2)
                if sd_cross_attention_maps2.max() >= max_attention_value:
                    max_attention_value = sd_cross_attention_maps2.max()
                    binarized_sd_cross_attention_maps2 = torch.where(
                        sd_cross_attention_maps2 > sd_cross_attention_maps2.mean() + 1 * sd_cross_attention_maps2.std(),
                        1, 0)
                    if torch.all(binarized_sd_cross_attention_maps2 == 0):
                        binarized_sd_cross_attention_maps2 = torch.where(
                            sd_cross_attention_maps2 > sd_cross_attention_maps2.mean(), 1, 0)
                    masked_pixels_ids = torch.where(binarized_sd_cross_attention_maps2.flatten() == 1)[0]
                    large_sd_self_attention_map = \
                        torch.nn.functional.interpolate(sd_self_attention_maps[0][None, ...],
                                                        crop_size, mode="bilinear")[0]
                    avg_large_sd_self_attention_map = (sd_cross_attention_maps2.flatten()[masked_pixels_ids][..., None, None] * large_sd_self_attention_map[masked_pixels_ids]).mean(dim=0)
                    avg_large_sd_self_attention_map = (avg_large_sd_self_attention_map - avg_large_sd_self_attention_map.min()) / (avg_large_sd_self_attention_map.max() - avg_large_sd_self_attention_map.min())

                    final_max_attention_map = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)
                    aux_max_attention_map = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)

                    final_max_attention_map[y_start:y_end, x_start:x_end] = (avg_large_sd_self_attention_map * sd_cross_attention_maps2.max())
                    aux_max_attention_map[y_start:y_end, x_start:x_end] = torch.ones_like(avg_large_sd_self_attention_map) * sd_cross_attention_maps2.max()

                if sd_cross_attention_maps2.max() >= threshold:
                    binarized_sd_cross_attention_maps2 = torch.where(
                        sd_cross_attention_maps2 > sd_cross_attention_maps2.mean() + 1 * sd_cross_attention_maps2.std(),
                        1, 0)
                    if torch.all(binarized_sd_cross_attention_maps2 == 0):
                        binarized_sd_cross_attention_maps2 = torch.where(
                            sd_cross_attention_maps2 > sd_cross_attention_maps2.mean(), 1, 0)
                    masked_pixels_ids = torch.where(binarized_sd_cross_attention_maps2.flatten() == 1)[0]
                    large_sd_self_attention_map = \
                        torch.nn.functional.interpolate(sd_self_attention_maps[0][None, ...],
                                                        crop_size, mode="bilinear")[0]
                    avg_large_sd_self_attention_map = (sd_cross_attention_maps2.flatten()[masked_pixels_ids][..., None, None] * large_sd_self_attention_map[masked_pixels_ids]).mean(dim=0)
                    avg_large_sd_self_attention_map = (avg_large_sd_self_attention_map - avg_large_sd_self_attention_map.min()) / (avg_large_sd_self_attention_map.max() - avg_large_sd_self_attention_map.min())

                    final_attention_map[y_start:y_end, x_start:x_end] += (avg_large_sd_self_attention_map * sd_cross_attention_maps2.max())
                    aux_attention_map[y_start:y_end, x_start:x_end] += torch.ones_like(avg_large_sd_self_attention_map) * sd_cross_attention_maps2.max()

        return final_attention_map, aux_attention_map, final_max_attention_map, aux_max_attention_map

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask_provided = not torch.all(mask == 0)
        if mask_provided:
            mask = mask[0]

        # small_final_attention_map, small_aux_attention_map, final_max_attention_map, aux_max_attention_map = self.get_patched_masks(image,
        #                                                                                           self.config.small_crop_size,
        #                                                                                           self.config.num_small_crops_per_side,
        #                                                                                           self.config.small_crop_threshold
        #                                                                                           )

        large_final_attention_map, large_aux_attention_map, final_max_attention_map, aux_max_attention_map = self.get_patched_masks(image,
                                                                                                  self.config.large_crop_size,
                                                                                                  self.config.num_large_crops_per_side,
                                                                                                  self.config.large_crop_threshold
                                                                                                  )

        if torch.all(large_final_attention_map == 0):
            final_max_attention_map /= aux_max_attention_map
            ones_indices = torch.where(aux_max_attention_map.flatten() > 0)[0]
            mean = final_max_attention_map.flatten()[ones_indices].mean()
            std = final_max_attention_map.flatten()[ones_indices].std()
            final_predicted_mask_0 = torch.where(final_max_attention_map > mean, 1, 0)
            final_predicted_mask_1 = torch.where(final_max_attention_map > mean + 1 * std, 1, 0)
            final_predicted_mask_2 = torch.where(final_max_attention_map > mean + 2 * std, 1, 0)
        else:
            large_final_attention_map /= large_aux_attention_map
            ones_indices = torch.where(large_aux_attention_map.flatten() > 0)[0]
            mean = large_final_attention_map.flatten()[ones_indices].mean()
            std = large_final_attention_map.flatten()[ones_indices].std()
            final_predicted_mask_0 = torch.where(large_final_attention_map > mean, 1, 0)
            final_predicted_mask_1 = torch.where(large_final_attention_map > mean + 1 * std, 1, 0)
            final_predicted_mask_2 = torch.where(large_final_attention_map > mean + 2 * std, 1, 0)
        final_predicted_mask_0 = final_predicted_mask_0.cpu()
        final_predicted_mask_1 = final_predicted_mask_1.cpu()
        final_predicted_mask_2 = final_predicted_mask_2.cpu()

        mask_grid_0 = torchvision.utils.make_grid(final_predicted_mask_0)
        mask_grid_1 = torchvision.utils.make_grid(final_predicted_mask_1)
        mask_grid_2 = torchvision.utils.make_grid(final_predicted_mask_2)

        masked_image_grid_0 = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - final_predicted_mask_0[None, ...]) + torch.stack(
                [final_predicted_mask_0 * 0, final_predicted_mask_0 * 0, final_predicted_mask_0], dim=0))
        masked_image_grid_1 = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - final_predicted_mask_1[None, ...]) + torch.stack(
                [final_predicted_mask_1 * 0, final_predicted_mask_1 * 0, final_predicted_mask_1], dim=0))
        masked_image_grid_2 = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - final_predicted_mask_2[None, ...]) + torch.stack(
                [final_predicted_mask_2 * 0, final_predicted_mask_2 * 0, final_predicted_mask_2], dim=0))

        image_grid = torchvision.utils.make_grid(image)

        self.logger.experiment.add_image("test predicted mask 0", mask_grid_0, batch_idx)
        self.logger.experiment.add_image("test predicted mask 1", mask_grid_1, batch_idx)
        self.logger.experiment.add_image("test predicted mask 2", mask_grid_2, batch_idx)
        self.logger.experiment.add_image("test masked image 0", masked_image_grid_0, batch_idx)
        self.logger.experiment.add_image("test masked image 1", masked_image_grid_1, batch_idx)
        self.logger.experiment.add_image("test masked image 2", masked_image_grid_2, batch_idx)
        self.logger.experiment.add_image("test image", image_grid, batch_idx)

        if mask_provided:
            if self.config.use_crf:
                crf_mask = torch.as_tensor(
                    crf((image[0].permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy().copy(
                        order='C'),
                        torch.stack([final_predicted_mask_0, final_predicted_mask_0,
                                     final_predicted_mask_0],
                                    dim=2).type(torch.float).numpy()))[:, :, 0]
                crf_masked_image_grid1 = torchvision.utils.make_grid(
                    image[0].cpu() * (1 - crf_mask[None, ...]).cpu() + crf_mask[None, ...])
                iou_0 = calculate_iou((crf_mask > 0).type(torch.uint8), (mask.cpu() > 0).type(torch.uint8))
            else:
                iou_0 = calculate_iou((final_predicted_mask_0 > 0).type(torch.uint8),
                                      (mask.cpu() > 0).type(torch.uint8))
                iou_1 = calculate_iou((final_predicted_mask_1 > 0).type(torch.uint8),
                                      (mask.cpu() > 0).type(torch.uint8))
                iou_2 = calculate_iou((final_predicted_mask_2 > 0).type(torch.uint8),
                                      (mask.cpu() > 0).type(torch.uint8))
            masks_grid = torchvision.utils.make_grid(mask[None, ...])

        if mask_provided:
            self.logger.experiment.add_image("test mask", masks_grid, batch_idx)
            self.log("test iou 0", iou_0, on_step=True, sync_dist=True)
            self.log("test iou 1", iou_1, on_step=True, sync_dist=True)
            self.log("test iou 2", iou_2, on_step=True, sync_dist=True)
            if self.config.use_crf:
                self.logger.experiment.add_image("test crf masked image1", crf_masked_image_grid1, batch_idx)

        return torch.tensor(0.)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            [
                # {'params': self.upsizer.parameters(), 'lr': self.config.lr_conv},
                {'params': self.token_t, 'lr': self.config.lr_2},
                {'params': self.token_u, 'lr': self.config.lr_1},

            ],
            lr=self.config.lr_1,
        )
        return {"optimizer": optimizer}

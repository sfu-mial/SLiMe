import math
import os

import pytorch_lightning as pl
import torch
import torchvision
from torch import optim

from src.config import Config
from src.crf import crf
from src.stable_difusion import StableDiffusion


def adjust_bbox_coords(long_side_length, short_side_length, short_side_coord_min, short_side_coord_max, short_side_original_length):
    margin = (long_side_length - short_side_length) / 2
    short_side_coord_min -= math.ceil(margin)
    short_side_coord_max += math.floor(margin)
    short_side_coord_max -= min(0, short_side_coord_min)
    short_side_coord_min = max(0, short_side_coord_min)
    short_side_coord_min += min(short_side_original_length - short_side_coord_max, 0)
    short_side_coord_max = min(short_side_original_length, short_side_coord_max)

    return short_side_coord_min, short_side_coord_max

class CoSegmenterTrainer(pl.LightningModule):
    def __init__(self, config: Config, learning_rate=0.001):
        super().__init__()
        self.counter = 0
        self.test_counter = 0
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.automatic_optimization = True
        self.stable_diffusion = StableDiffusion(sd_version='1.4', partial_run=False, attention_layers_to_use=config.attention_layers_to_use)

        self.target_image_original = None
        self.final_prediction1 = 0
        self.final_prediction2 = 0

        self.aux_final_prediction1 = 0
        self.aux_final_prediction2 = 0

        self.min_loss_1 = 10000000
        self.min_loss_2 = 10000000
        self.generate_noise = True

        self.first_binarized_attention_map = None

        if self.config.train:
            uncond_embeddings, text_embeddings = self.stable_diffusion.get_text_embeds("dog eyes", "dog eyes")
            self.token_t = text_embeddings[:, 2:3].clone()
            self.token_t.requires_grad_(True)
            self.text_embedding = torch.cat([text_embeddings[:, :2], self.token_t, text_embeddings[:, 3:]], dim=1)

            self.token_u = uncond_embeddings[:, 2:3].clone()
            self.token_u.requires_grad_(True)
            self.uncond_embedding = torch.cat([uncond_embeddings[:, :2], self.token_u, uncond_embeddings[:, 3:]], dim=1)

    def on_fit_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))

    def get_attention_map(self, raw_attention_maps, output_size=256, token_id=2, use_softmax=True):
        all_processed_attention_maps1 = []
        all_processed_attention_maps2 = []
        for layer in raw_attention_maps:
            # if layer.startswith("up_blocks[1]"):
            channel, img_embed_len, text_embed_len = raw_attention_maps[layer].chunk(2)[1].shape
            if use_softmax:
                reshaped_attention_map = raw_attention_maps[layer].chunk(2)[1].softmax(dim=-1)[:, :, token_id].reshape(
                    channel,
                    1,
                    int(math.sqrt(img_embed_len)),
                    int(math.sqrt(img_embed_len)),
                )
            else:
                reshaped_attention_map = raw_attention_maps[layer].chunk(2)[1][:, :, token_id].reshape(
                    channel,
                    1,
                    int(math.sqrt(img_embed_len)),
                    int(math.sqrt(img_embed_len)),
                )
            resized_attention_maps1 = \
            torch.nn.functional.interpolate(reshaped_attention_map, size=output_size,
                                            mode='bilinear').mean(dim=0)[0]
            all_processed_attention_maps1.append(resized_attention_maps1)

            if layer.startswith("up_blocks[2]"):
                channel, img_embed_len, text_embed_len = raw_attention_maps[layer].chunk(2)[0].shape
                if use_softmax:
                    reshaped_attention_map = raw_attention_maps[layer].chunk(2)[0].softmax(dim=-1)[:, :, token_id].reshape(
                        channel,
                        1,
                        int(math.sqrt(img_embed_len)),
                        int(math.sqrt(img_embed_len)),
                    )
                else:
                    reshaped_attention_map = raw_attention_maps[layer].chunk(2)[0][:, :, token_id].reshape(
                        channel,
                        1,
                        int(math.sqrt(img_embed_len)),
                        int(math.sqrt(img_embed_len)),
                    )
                resized_attention_maps2 = \
                torch.nn.functional.interpolate(reshaped_attention_map, size=output_size,
                                                mode='bilinear').mean(dim=0)[0]

                all_processed_attention_maps2.append(resized_attention_maps2)

        sd_attention_maps1 = torch.stack(all_processed_attention_maps1, dim=0).mean(dim=0)
        sd_attention_maps2 = torch.stack(all_processed_attention_maps2, dim=0).mean(dim=0)
        if not use_softmax:
            sd_attention_maps1 = torch.sigmoid(sd_attention_maps1)
            sd_attention_maps2 = torch.sigmoid(sd_attention_maps2)

        return sd_attention_maps1.to(self.device), sd_attention_maps2.to(self.device)

    def training_step(self, batch, batch_idx):
        src_images, mask1, mask2 = batch
        mask1 = mask1[0]
        mask2 = mask2[0]
        self.text_embedding = torch.cat([self.text_embedding[:, :2], self.token_t, self.text_embedding[:, 3:]], dim=1)
        self.uncond_embedding = torch.cat([self.uncond_embedding[:, :2], self.token_u, self.uncond_embedding[:, 3:]],
                                          dim=1)

        loss, raw_attention_maps = self.stable_diffusion.train_step(
            torch.repeat_interleave(torch.cat([self.uncond_embedding, self.text_embedding]), self.config.batch_size, 0),
            src_images, t=torch.tensor(260),
            back_propagate_loss=False, generate_new_noise=self.generate_noise)
        self.generate_noise = False
        sd_attention_maps1, sd_attention_maps2 = self.get_attention_map(
            raw_attention_maps=raw_attention_maps,
            output_size=self.config.mask_size,
            token_id=2,
            use_softmax=self.config.train_use_softmax,
        )

        loss1 = torch.nn.functional.mse_loss(sd_attention_maps1, mask1)
        loss2 = torch.nn.functional.mse_loss(sd_attention_maps2, mask2)

        if loss1 < self.min_loss_1:
            self.min_loss_1 = loss1
            torch.save(self.text_embedding, os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_1.pth"))
        if loss2 < self.min_loss_2:
            self.min_loss_2 = loss2
            torch.save(self.uncond_embedding, os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_2.pth"))

        loss = loss1 + loss2

        self.log("mse_loss1", loss1, on_step=True, sync_dist=True)
        self.log("mse_loss2", loss2, on_step=True, sync_dist=True)

        images_grid = torchvision.utils.make_grid(torch.nn.functional.interpolate(src_images, size=mask1.shape))
        sd_attention_maps1 = (sd_attention_maps1 - sd_attention_maps1.min()) / (
                sd_attention_maps1.max() - sd_attention_maps1.min())
        sd_attention_maps2 = (sd_attention_maps2 - sd_attention_maps2.min()) / (
                sd_attention_maps2.max() - sd_attention_maps2.min())
        sd_attention_maps_grid1 = torchvision.utils.make_grid(sd_attention_maps1[None, ...])
        sd_attention_maps_grid2 = torchvision.utils.make_grid(sd_attention_maps2[None, ...])

        mask1_grid = torchvision.utils.make_grid(mask1[None, ...])
        mask2_grid = torchvision.utils.make_grid(mask2[None, ...])

        self.logger.experiment.add_image("train image", images_grid, 0)
        self.logger.experiment.add_image("train sd attention maps1", sd_attention_maps_grid1, self.counter)
        self.logger.experiment.add_image("train sd attention maps2", sd_attention_maps_grid2, self.counter)
        self.logger.experiment.add_image("train mask1", mask1_grid, self.counter)
        self.logger.experiment.add_image("train mask2", mask2_grid, self.counter)
        self.counter += 1

        return loss

    def on_train_end(self) -> None:
        torch.save(self.stable_diffusion.noise, os.path.join(self.config.checkpoint_dir, "noise.pth"))

    def on_test_epoch_start(self) -> None:
        self.text_embedding = torch.load(os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_1.pth")).to(
            self.device)
        self.uncond_embedding = torch.load(
            os.path.join(self.config.checkpoint_dir, "optimized_text_embedding_2.pth")).to(
            self.device)
        noise = torch.load(os.path.join(self.config.checkpoint_dir, "noise.pth"))
        self.stable_diffusion.noise = noise

    def test_step(self, batch, batch_idx):
        target_images, crop_start_y, crop_start_x, mask_original, target_images_original = batch
        target_images_original = target_images_original[0]
        mask_provided = not torch.all(mask_original == 0)
        if mask_provided:
            mask_original = mask_original[0]

        with torch.no_grad():
            loss, raw_attention_maps = self.stable_diffusion.train_step(
                torch.cat([self.uncond_embedding, self.text_embedding], dim=0), target_images,
                t=torch.tensor(160), back_propagate_loss=False, generate_new_noise=False)
        sd_attention_maps1, sd_attention_maps2 = self.get_attention_map(
            raw_attention_maps=raw_attention_maps,
            output_size=512,
            token_id=2,
            use_softmax=self.config.test_use_softmax,
        )
        sd_attention_maps1 = (sd_attention_maps1 - sd_attention_maps1.min()) / (
                sd_attention_maps1.max() - sd_attention_maps1.min())
        sd_attention_maps2 = (sd_attention_maps2 - sd_attention_maps2.min()) / (
                sd_attention_maps2.max() - sd_attention_maps2.min())

        original_size_attention_map = torch.zeros(int(512 / self.config.test_crop_ratio),
                                                  int(512 / self.config.test_crop_ratio))
        aux_original_size_attention_map = torch.zeros(int(512 / self.config.test_crop_ratio),
                                                      int(512 / self.config.test_crop_ratio)) + 1e-7

        original_size_attention_map[crop_start_y[0]:crop_start_y[0] + 512,
        crop_start_x[0]:crop_start_x[0] + 512] = sd_attention_maps1
        aux_original_size_attention_map[crop_start_y[0]:crop_start_y[0] + 512,
        crop_start_x[0]:crop_start_x[0] + 512] = torch.ones_like(sd_attention_maps1)

        self.final_prediction1 += original_size_attention_map
        self.aux_final_prediction1 += aux_original_size_attention_map

        original_size_attention_map = torch.zeros(int(512 / self.config.test_crop_ratio),
                                                  int(512 / self.config.test_crop_ratio))
        aux_original_size_attention_map = torch.zeros(int(512 / self.config.test_crop_ratio),
                                                      int(512 / self.config.test_crop_ratio)) + 1e-7

        original_size_attention_map[crop_start_y[0]:crop_start_y[0] + 512,
        crop_start_x[0]:crop_start_x[0] + 512] = sd_attention_maps2
        aux_original_size_attention_map[crop_start_y[0]:crop_start_y[0] + 512,
        crop_start_x[0]:crop_start_x[0] + 512] = torch.ones_like(sd_attention_maps2)

        self.final_prediction2 += original_size_attention_map
        self.aux_final_prediction2 += aux_original_size_attention_map

        self.test_counter += 1

        if self.test_counter == self.config.test_num_crops:
            self.final_prediction1 /= self.aux_final_prediction1
            self.final_prediction2 /= self.aux_final_prediction2
            final_prediction3 = (self.final_prediction1 + self.final_prediction2) / 2
            if self.config.threshold1 == "mean+std":
                binarized_attention_map1 = torch.where(
                    self.final_prediction1 > torch.mean(self.final_prediction1) + 1 * torch.std(self.final_prediction1),
                    1,
                    0)
            elif isinstance(self.config.threshold1, float):
                binarized_attention_map1 = torch.where(self.final_prediction1 > self.config.threshold1, 1., 0.)
                binarized_attention_map3 = torch.where(final_prediction3 > self.config.threshold1, 1., 0.)
            if self.config.threshold2 == "mean+std":
                binarized_attention_map2 = torch.where(
                    self.final_prediction2 > torch.mean(self.final_prediction2) + 1 * torch.std(self.final_prediction2),
                    1,
                    0)
            elif isinstance(self.config.threshold2, float):
                binarized_attention_map2 = torch.where(self.final_prediction2 > self.config.threshold2, 1., 0.)
            ys, xs = torch.where(binarized_attention_map1 == 1)
            x_start, x_end, y_start, y_end = xs.min(), xs.max(), ys.min(), ys.max()
            w, h = x_end - x_start, y_end - y_start
            if w < h:
                diff = max(512 // 4, h) - h
                offset_start, offset_end = math.ceil(diff / 2), math.floor(diff / 2)
                if y_start - math.ceil(diff / 2) < 0:
                    offset_end -= y_start - math.ceil(diff / 2)
                    offset_start = y_start
                elif y_end + math.floor(diff / 2) > 511:
                    offset_start += y_end + math.floor(diff / 2) - 511
                    offset_end = 511 - y_end
                y_start -= offset_start
                y_end += offset_end

            elif h < w:
                diff = max(512 // 4, w) - w
                offset_start, offset_end = math.ceil(diff / 2), math.floor(diff / 2)
                if x_start - diff // 2 < 0:
                    offset_end -= x_start - diff // 2
                    offset_start = x_start
                elif x_end + math.floor(diff / 2) > 511:
                    offset_start += x_end + math.floor(diff / 2) - 511
                    offset_end = 511 - x_end
                x_start -= offset_start
                x_end += offset_end

            w, h = x_end - x_start, y_end - y_start
            if w > h:
                y_start, y_end = adjust_bbox_coords(w, h, y_start, y_end, 512)
            elif w < h:
                x_start, x_end = adjust_bbox_coords(h, w, x_start, x_end, 512)
            crop_size = max(h, w)
            target_images = target_images[:, :, y_start:y_end, x_start:x_end]
            target_images = torch.nn.functional.interpolate(target_images, 512, mode="bilinear")
            with torch.no_grad():
                loss, raw_attention_maps = self.stable_diffusion.train_step(
                    torch.cat([self.uncond_embedding, self.text_embedding], dim=0), target_images,
                    t=torch.tensor(160), back_propagate_loss=False, generate_new_noise=False)
            sd_attention_maps1, sd_attention_maps2 = self.get_attention_map(
                raw_attention_maps=raw_attention_maps,
                output_size=512,
                token_id=2,
                use_softmax=self.config.test_use_softmax,
            )
            sd_attention_maps1, sd_attention_maps2 = torch.nn.functional.interpolate(sd_attention_maps1[None, None, ...], crop_size, mode="bilinear")[0, 0], torch.nn.functional.interpolate(sd_attention_maps2[None, None, ...], crop_size, mode="bilinear")[0, 0]

            sd_attention_maps1 = (sd_attention_maps1 - sd_attention_maps1.min()) / (
                    sd_attention_maps1.max() - sd_attention_maps1.min())
            sd_attention_maps2 = (sd_attention_maps2 - sd_attention_maps2.min()) / (
                    sd_attention_maps2.max() - sd_attention_maps2.min())

            original_size_attention_map1 = torch.zeros(512, 512)

            original_size_attention_map1[y_start: y_end, x_start: x_end] = sd_attention_maps1

            original_size_attention_map2 = torch.zeros(512, 512)

            original_size_attention_map2[y_start: y_end, x_start: x_end] = sd_attention_maps2

            new_binarized_attention_map1 = torch.where(original_size_attention_map1 > self.config.threshold1, 1., 0.)
            new_binarized_attention_map2 = torch.where(original_size_attention_map2 > self.config.threshold2, 1., 0.)

            if mask_provided:
                if self.config.use_crf:
                    crf_mask = torch.as_tensor(
                        crf((target_images_original.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy().copy(order='C'),
                            torch.stack([binarized_attention_map1, binarized_attention_map1, binarized_attention_map1],
                                        dim=2).type(torch.float).numpy()))[:, :, 0]
                    crf_masked_image_grid1 = torchvision.utils.make_grid(
                        target_images_original.cpu() * (1 - crf_mask[None, ...]).cpu() + crf_mask[None, ...])
                    intersection = (torch.nn.functional.interpolate(crf_mask[None, None, ...],
                                                                    self.config.mask_size)[0, 0] > 0).type(torch.uint8) * (
                                           mask_original.cpu() > 0).type(torch.uint8)
                    union = (torch.nn.functional.interpolate(crf_mask[None, None, ...],
                                                             self.config.mask_size)[0, 0] > 0).type(torch.uint8) + (
                                    mask_original.cpu() > 0).type(torch.uint8) - intersection
                else:
                    intersection = (torch.nn.functional.interpolate(binarized_attention_map1[None, None, ...], self.config.mask_size)[0, 0]>0).type(torch.uint8) * (mask_original.cpu()>0).type(torch.uint8)
                    union = (torch.nn.functional.interpolate(binarized_attention_map1[None, None, ...], self.config.mask_size)[0, 0]>0).type(torch.uint8) + (mask_original.cpu()>0).type(torch.uint8) - intersection
                iou = intersection.sum() / (union.sum()+1e-7)
                masks_grid = torchvision.utils.make_grid(mask_original[None, ...])

            images_grid = torchvision.utils.make_grid(target_images_original)

            masked_image_grid1 = torchvision.utils.make_grid(
                target_images_original.cpu() * (1 - binarized_attention_map1[None, ...]).cpu() + torch.stack(
                    [binarized_attention_map1 * 0, binarized_attention_map1 * 0, binarized_attention_map1], dim=0))
            masked_image_grid2 = torchvision.utils.make_grid(
                target_images_original.cpu() * (1 - binarized_attention_map2[None, ...]).cpu() + torch.stack(
                    [binarized_attention_map2 * 0, binarized_attention_map2 * 0, binarized_attention_map2], dim=0))

            new_masked_image_grid1 = torchvision.utils.make_grid(
                target_images_original.cpu() * (1 - new_binarized_attention_map1[None, ...]).cpu() + torch.stack(
                    [new_binarized_attention_map1 * 0, new_binarized_attention_map1 * 0, new_binarized_attention_map1], dim=0))
            new_masked_image_grid2 = torchvision.utils.make_grid(
                target_images_original.cpu() * (1 - new_binarized_attention_map2[None, ...]).cpu() + torch.stack(
                    [new_binarized_attention_map2 * 0, new_binarized_attention_map2 * 0, new_binarized_attention_map2], dim=0))

            masked_image_grid3 = torchvision.utils.make_grid(
                target_images_original.cpu() * (1 - binarized_attention_map3[None, ...]).cpu() + torch.stack(
                    [binarized_attention_map3 * 0, binarized_attention_map3 * 0, binarized_attention_map3], dim=0))

            attention_grid1 = torchvision.utils.make_grid(self.final_prediction1)
            attention_grid2 = torchvision.utils.make_grid(self.final_prediction2)
            new_attention_grid1 = torchvision.utils.make_grid(original_size_attention_map1)
            new_attention_grid2 = torchvision.utils.make_grid(original_size_attention_map2)
            attention_grid3 = torchvision.utils.make_grid((self.final_prediction1+self.final_prediction2)/2)

            log_id = batch_idx // self.config.test_num_crops

            self.logger.experiment.add_image("test image", images_grid, log_id)

            if mask_provided:
                self.logger.experiment.add_image("test mask", masks_grid, log_id)
                self.log("test iou", iou, on_step=True, sync_dist=True)
                if self.config.use_crf:
                    self.logger.experiment.add_image("test crf masked image1", crf_masked_image_grid1, log_id)

            self.logger.experiment.add_image("test masked image1", masked_image_grid1, log_id)
            self.logger.experiment.add_image("test masked image2", masked_image_grid2, log_id)
            self.logger.experiment.add_image("test new masked image1", new_masked_image_grid1, log_id)
            self.logger.experiment.add_image("test new masked image2", new_masked_image_grid2, log_id)
            self.logger.experiment.add_image("test masked image3", masked_image_grid3, log_id)
            self.logger.experiment.add_image("test attention maps1", attention_grid1, log_id)
            self.logger.experiment.add_image("test attention maps2", attention_grid2, log_id)
            self.logger.experiment.add_image("test new attention maps1", new_attention_grid1, log_id)
            self.logger.experiment.add_image("test new attention maps2", new_attention_grid2, log_id)
            self.logger.experiment.add_image("test attention maps3", attention_grid3, log_id)

            self.final_prediction1 = 0
            self.final_prediction2 = 0
            self.aux_final_prediction1 = 0
            self.aux_final_prediction2 = 0
        self.test_counter %= self.config.test_num_crops
        return loss

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            [
                {'params': self.token_t, 'lr': self.config.lr_1},
                {'params': self.token_u, 'lr': self.config.lr_2},

            ],
            lr=self.config.lr_1,
        )
        return {"optimizer": optimizer}

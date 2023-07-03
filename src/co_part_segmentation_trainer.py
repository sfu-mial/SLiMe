import os

import pytorch_lightning as pl
import torch
import torchvision
from torch import optim

from src.config import Config
from src.stable_difusion import StableDiffusion
from src.utils import calculate_iou, get_crops_coords


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
        self.epoch_loss_1 = []
        self.epoch_loss_2 = []
        self.generate_noise = True
        os.makedirs(self.config.train_checkpoint_dir, exist_ok=True)

        # self.uncond_embedding, self.text_embedding = self.stable_diffusion.get_text_embeds("object part", "object part")

        # if self.config.train:
            # uncond_embeddings, text_embeddings = self.stable_diffusion.get_text_embeds("object part", "object part")
            # self.token_t = text_embeddings[:, 2:3].clone()
            # self.token_t.requires_grad_(True)
            # self.text_embedding = torch.cat([text_embeddings[:, :2], self.token_t, text_embeddings[:, 3:]], dim=1)
            #
            # self.token_u = uncond_embeddings[:, 2:3].clone()
            # self.token_u.requires_grad_(True)
            # self.uncond_embedding = torch.cat([uncond_embeddings[:, :2], self.token_u, uncond_embeddings[:, 3:]], dim=1)
        self.ious = {}
        for part_name in self.config.test_part_names:
            self.ious[part_name] = []

        self.translator = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768)
        )

    def on_train_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))
        self.uncond_embedding, self.text_embedding = self.stable_diffusion.get_text_embeds("object part", "object part")

    def training_step(self, batch, batch_idx):
        src_images, mask1, mask2 = batch
        small_mask1 = torch.nn.functional.interpolate(mask1[None, ...], 64, mode="nearest")[0]
        mask1 = mask1[0]
        mask2 = mask2[0]
        small_mask1 = small_mask1[0]
        # self.text_embedding = torch.cat([self.text_embedding[:, :2], self.token_t, self.text_embedding[:, 3:]], dim=1)
        # self.uncond_embedding = torch.cat([self.uncond_embedding[:, :2], self.token_u, self.uncond_embedding[:, 3:]],
        #                                   dim=1)
        loss, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps = self.stable_diffusion.train_step(
            torch.repeat_interleave(torch.cat([self.uncond_embedding, self.translator(self.text_embedding)]), self.config.batch_size, 0),
            src_images, t=torch.tensor(20),
            back_propagate_loss=False, generate_new_noise=self.generate_noise,
            attention_output_size=self.config.mask_size, token_ids=list(range(77)), train=True, average_layers=False, apply_softmax=False)
        self.generate_noise = False
        loss3 = 0
        self_attention_maps = []
        if sd_self_attention_maps is not None:
            for i in range(len(self.config.train_part_names)):
                self_attention_map = sd_self_attention_maps[0][torch.where(small_mask1.flatten() == i)[0]].mean(dim=0)
                loss3 = loss3 + torch.nn.functional.mse_loss(self_attention_map, mask1)
                self_attention_maps.append(((self_attention_map - self_attention_map.min()) / (self_attention_map.max() - self_attention_map.min())).detach().cpu())
        if len(self_attention_maps) > 0:
            self_attention_maps = torch.stack(self_attention_maps, dim=0)
        # sd_cross_attention_maps1 = (sd_cross_attention_maps1 - sd_cross_attention_maps1.min()) / (
        #         sd_cross_attention_maps1.max() - sd_cross_attention_maps1.min())
        # sd_cross_attention_maps2 = (sd_cross_attention_maps2 - sd_cross_attention_maps2.min()) / (
        #         sd_cross_attention_maps2.max() - sd_cross_attention_maps2.min())
        loss1 = torch.tensor(0.)
        loss2 = torch.tensor(0.)
        for layer_sd_cross_attention_maps2 in sd_cross_attention_maps2:
            loss2 = loss2 + torch.nn.functional.cross_entropy(layer_sd_cross_attention_maps2[None, ...], mask1[None, ...].type(torch.long))
        loss2 = loss2 / sd_cross_attention_maps2.shape[0]
        sd_cross_attention_maps2 = torch.unsqueeze(sd_cross_attention_maps2.mean(dim=0).softmax(dim=0), dim=1)
        sd_cross_attention_maps1 = torch.unsqueeze(sd_cross_attention_maps1.mean(dim=0).softmax(dim=0), dim=1)
        # loss1 = torch.nn.functional.mse_loss(sd_cross_attention_maps1[0], mask1)
        # loss2 = torch.nn.functional.mse_loss(sd_cross_attention_maps2[0], mask2)
        # loss1 = torch.nn.functional.mse_loss(sd_cross_attention_maps1[0], mask2)
        # loss2 = torch.nn.functional.mse_loss(sd_cross_attention_maps2[0], mask1)
        self.epoch_loss_1.append(loss1.detach().item())
        self.epoch_loss_2.append(loss2.detach().item())
        # if loss1 < self.min_loss_1:
        #     self.min_loss_1 = loss1.detach().item()
        #     torch.save(self.text_embedding[0, 2], os.path.join(self.config.train_checkpoint_dir, "optimized_text_embedding_1.pth"))
        # if loss2 < self.min_loss_2:
        #     self.min_loss_2 = loss2.detach().item()
        #     # torch.save(self.uncond_embedding[0, 2],
        #     #            os.path.join(self.config.train_checkpoint_dir, "optimized_text_embedding_2.pth"))
        #     torch.save(self.translator.state_dict(),
        #                os.path.join(self.config.train_checkpoint_dir, "translator.pth"))

        loss = loss1 + loss2 + loss3

        self.log("mse_loss1", loss1.detach().cpu(), on_step=True, sync_dist=True)
        self.log("mse_loss2", loss2.detach().cpu(), on_step=True, sync_dist=True)
        self.log("mse_loss3", loss3.detach().cpu(), on_step=True, sync_dist=True)

        sd_cross_attention_maps2 = (sd_cross_attention_maps2 - sd_cross_attention_maps2.min()) / (
                sd_cross_attention_maps2.max() - sd_cross_attention_maps2.min())
        # sd_cross_attention_maps1 = (sd_cross_attention_maps1 - sd_cross_attention_maps1.min()) / (
        #         sd_cross_attention_maps1.max() - sd_cross_attention_maps1.min())

        if sd_self_attention_maps is not None:
            # self_attention_map = (self_attention_map - self_attention_map.min()) / (self_attention_map.max() - self_attention_map.min())
            self_attention_map_grid = torchvision.utils.make_grid(self_attention_maps.unsqueeze(dim=1))
            self.logger.experiment.add_image("train self attention map", self_attention_map_grid, self.counter)

        images_grid = torchvision.utils.make_grid(src_images.detach().cpu())
        sd_attention_maps_grid1 = torchvision.utils.make_grid(sd_cross_attention_maps1.detach().cpu())
        sd_attention_maps_grid2 = torchvision.utils.make_grid(sd_cross_attention_maps2.detach().cpu())
        # print(mask1[None, ...].shape, mask1.unique())
        mask1_grid = torchvision.utils.make_grid(mask1[None, ...].detach().cpu()/len(self.config.train_part_names)*255)
        mask2_grid = torchvision.utils.make_grid(mask2[None, ...].detach().cpu())

        self.logger.experiment.add_image("train image", images_grid, self.counter)
        self.logger.experiment.add_image("train sd attention maps1", sd_attention_maps_grid1, self.counter)
        self.logger.experiment.add_image("train sd attention maps2", sd_attention_maps_grid2, self.counter)
        self.logger.experiment.add_image("train mask1", mask1_grid, self.counter)
        self.logger.experiment.add_image("train mask2", mask2_grid, self.counter)
        self.counter += 1

        return loss

    def on_train_epoch_end(self) -> None:
        mean_epoch_loss_1 = sum(self.epoch_loss_1) / len(self.epoch_loss_1)
        mean_epoch_loss_2 = sum(self.epoch_loss_2) / len(self.epoch_loss_2)
        if mean_epoch_loss_1 < self.min_loss_1:
            self.min_loss_1 = mean_epoch_loss_1
            torch.save(self.text_embedding[0, 2], os.path.join(self.config.train_checkpoint_dir, "optimized_text_embedding_1.pth"))
        if mean_epoch_loss_2 < self.min_loss_2:
            self.min_loss_2 = mean_epoch_loss_2
            # torch.save(self.uncond_embedding[0, 2],
            #            os.path.join(self.config.train_checkpoint_dir, "optimized_text_embedding_2.pth"))
            torch.save(self.translator.state_dict(),
                       os.path.join(self.config.train_checkpoint_dir, "translator.pth"))
        # torch.save(self.translator.state_dict(),
        #            os.path.join(self.config.train_checkpoint_dir, f"translator_{self.current_epoch}.pth"))

    def on_train_end(self) -> None:
        torch.save(self.stable_diffusion.noise, os.path.join(self.config.train_checkpoint_dir, "noise.pth"))

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))
        self.stable_diffusion.change_hooks(attention_layers_to_use=self.config.attention_layers_to_use)  # exclude self attention layer
        self.uncond_embedding, self.text_embedding = self.stable_diffusion.get_text_embeds("object part", "object part")
        self.translator.load_state_dict(torch.load(os.path.join(self.config.test_checkpoint_dirs[0], "translator.pth")))

        # for i, checkpoint_dir in enumerate(self.config.test_checkpoint_dirs):
        #     text_embedding = torch.load(
        #         os.path.join(checkpoint_dir, "optimized_text_embedding_1.pth")).to(
        #         self.device)
        #     if len(text_embedding.shape) == 3:  # to support previously optimized embeddings
        #         text_embedding = text_embedding[0, 2]
        #     uncond_embedding = torch.load(
        #         os.path.join(checkpoint_dir, "optimized_text_embedding_2.pth")).to(
        #         self.device)
        #     if len(uncond_embedding.shape) == 3:  # to support previously optimized embeddings
        #         uncond_embedding = uncond_embedding[0, 2]
        #     self.text_embedding[0, i + 1] = text_embedding
        #     self.uncond_embedding[0, i + 1] = uncond_embedding
        #     noise = torch.load(os.path.join(self.config.train_checkpoint_dir, "noise.pth")).to(self.device)
        #     self.stable_diffusion.noise = noise

    def get_patched_masks(self, image, crop_size, num_crops_per_side, threshold):
        crops_coords = get_crops_coords(image.shape[2:], crop_size,
                                        num_crops_per_side)

        final_attention_map = torch.zeros(len(self.config.test_part_names), image.shape[2], image.shape[3])
        aux_attention_map = torch.zeros(len(self.config.test_part_names), image.shape[2], image.shape[3], dtype=torch.uint8) + 1e-7

        for i in range(num_crops_per_side):
            for j in range(num_crops_per_side):
                y_start, y_end, x_start, x_end = crops_coords[i * num_crops_per_side + j]
                cropped_image = image[:, :, y_start:y_end, x_start:x_end]
                # cropped_image = torch.nn.functional.interpolate(cropped_image, 512, mode="bilinear")
                with torch.no_grad():
                    loss, _, sd_cross_attention_maps2, sd_self_attention_maps = self.stable_diffusion.train_step(
                        torch.cat([self.uncond_embedding, self.translator(self.text_embedding)], dim=0), cropped_image,
                        t=torch.tensor(10), back_propagate_loss=False, generate_new_noise=True,
                        attention_output_size=64,
                        token_ids=list(range(len(self.config.test_part_names))), train=False)
                self.stable_diffusion.attention_maps = {}
                self_attention_map = \
                    torch.nn.functional.interpolate(sd_self_attention_maps[0][None, ...],
                                                    crop_size, mode="bilinear")[0].detach()

                # self_attention_map : 64x64, 64, 64
                # sd_cross_attention_maps2 : len(self.config.test_checkpoint_dirs), 64, 64

                attention_maps = sd_cross_attention_maps2.flatten(1, 2).detach()  # len(self.config.test_checkpoint_dirs) , 64x64
                max_values = attention_maps.max(dim=1).values  # len(self.config.test_checkpoint_dirs)
                min_values = attention_maps.min(dim=1).values  # len(self.config.test_checkpoint_dirs)
                passed_indices = torch.where(max_values >= threshold)[0]  #
                # not_passed_indices = torch.where(max_values < threshold)[0]
                if len(passed_indices) > 0:
                    passed_attention_maps = attention_maps[passed_indices]
                    passed_masks = torch.where(passed_attention_maps > passed_attention_maps.mean(1, keepdim=True) + passed_attention_maps.std(1, keepdim=True), 1, 0)
                    zero_masks_indices = torch.all(passed_masks == 0, dim=1)
                    if zero_masks_indices.sum() > 0:  # there is at least one mask with all values equal to zero
                        passed_masks[zero_masks_indices] = torch.where(
                            passed_attention_maps[zero_masks_indices] > passed_attention_maps.mean(1, keepdim=True)[zero_masks_indices], 1, 0)
                    for idx, mask_id in enumerate(passed_indices):
                        masked_pixels_ids = torch.where(passed_masks[idx] == 1)[0]
                        avg_self_attention_map = (
                                    passed_attention_maps[idx, masked_pixels_ids][..., None, None] *
                                    self_attention_map[masked_pixels_ids]).mean(dim=0)
                        avg_self_attention_map_min = avg_self_attention_map.min()
                        avg_self_attention_map_max = avg_self_attention_map.max()
                        coef = (avg_self_attention_map_max - avg_self_attention_map_min) / (
                                    max_values[mask_id] - min_values[mask_id])
                        # avg_self_attention_map = (avg_self_attention_map - avg_self_attention_map.min()) / (avg_self_attention_map.max() - avg_self_attention_map.min())
                        final_attention_map[mask_id, y_start:y_end, x_start:x_end] += (
                                (avg_self_attention_map / coef) + (
                                    min_values[mask_id] - avg_self_attention_map_min / coef)).cpu()
                        # final_attention_map[mask_id, y_start:y_end, x_start:x_end] += (
                        #             avg_self_attention_map * max_values[mask_id])
                        aux_attention_map[mask_id, y_start:y_end, x_start:x_end] += (torch.ones_like(avg_self_attention_map, dtype=torch.uint8)).cpu()

                # if len(not_passed_indices) > 0:
                #     not_passed_attention_maps = attention_maps[not_passed_indices]
                #     greater_than_max_not_passed_indices = torch.where(max_values[not_passed_indices] > max_attention_value[not_passed_indices])[0]
                #     if len(greater_than_max_not_passed_indices) > 0:
                #         max_attention_value[not_passed_indices[greater_than_max_not_passed_indices]] = max_values[
                #             not_passed_indices[greater_than_max_not_passed_indices]]
                #         greater_than_max_not_passed_attention_maps = not_passed_attention_maps[greater_than_max_not_passed_indices]
                #         greater_than_max_not_passed_masks = torch.where(
                #             greater_than_max_not_passed_attention_maps > greater_than_max_not_passed_attention_maps.mean(1, keepdim=True) + greater_than_max_not_passed_attention_maps.std(1, keepdim=True),
                #             1, 0)
                #         zero_masks_indices = torch.all(greater_than_max_not_passed_masks == 0, dim=1)
                #         if zero_masks_indices.sum() > 0:  # there is at least one mask with all values equal to zero
                #             greater_than_max_not_passed_masks[zero_masks_indices] = torch.where(
                #                 greater_than_max_not_passed_attention_maps[zero_masks_indices] > greater_than_max_not_passed_attention_maps.mean(1, keepdim=True)[
                #                     zero_masks_indices], 1, 0)
                #         for idx, mask_id in enumerate(greater_than_max_not_passed_indices):
                #             masked_pixels_ids = torch.where(greater_than_max_not_passed_masks[idx] == 1)[0]
                #             avg_self_attention_map = (
                #                     greater_than_max_not_passed_attention_maps[idx, masked_pixels_ids][..., None, None] *
                #                     self_attention_map[masked_pixels_ids]).mean(dim=0)
                #             avg_self_attention_map = (avg_self_attention_map - avg_self_attention_map.min()) / (
                #                         avg_self_attention_map.max() - avg_self_attention_map.min())
                #
                #             final_max_attention_map_ = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)
                #
                #             final_max_attention_map_[y_start:y_end, x_start:x_end] = avg_self_attention_map * max_values[mask_id]
                #
                #             final_max_attention_map[not_passed_indices[mask_id]] = final_max_attention_map_


                # if sd_cross_attention_maps2.max() >= max_attention_value:
                #     max_attention_value = sd_cross_attention_maps2.max()
                #     binarized_sd_cross_attention_maps2 = torch.where(
                #         sd_cross_attention_maps2 > sd_cross_attention_maps2.mean() + 1 * sd_cross_attention_maps2.std(),
                #         1, 0)
                #     if torch.all(binarized_sd_cross_attention_maps2 == 0):
                #         binarized_sd_cross_attention_maps2 = torch.where(
                #             sd_cross_attention_maps2 > sd_cross_attention_maps2.mean(), 1, 0)
                #     masked_pixels_ids = torch.where(binarized_sd_cross_attention_maps2.flatten() == 1)[0]
                #     large_sd_self_attention_map = \
                #         torch.nn.functional.interpolate(sd_self_attention_maps[0][None, ...],
                #                                         crop_size, mode="bilinear")[0]
                #     avg_large_sd_self_attention_map = (sd_cross_attention_maps2.flatten()[masked_pixels_ids][..., None, None] * large_sd_self_attention_map[masked_pixels_ids]).mean(dim=0)
                #     avg_large_sd_self_attention_map = (avg_large_sd_self_attention_map - avg_large_sd_self_attention_map.min()) / (avg_large_sd_self_attention_map.max() - avg_large_sd_self_attention_map.min())
                #
                #     final_max_attention_map = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)
                #     aux_max_attention_map = torch.zeros(*image.shape[2:]).to(self.stable_diffusion.device)
                #
                #     final_max_attention_map[y_start:y_end, x_start:x_end] = (avg_large_sd_self_attention_map * sd_cross_attention_maps2.max())
                #     aux_max_attention_map[y_start:y_end, x_start:x_end] = torch.ones_like(avg_large_sd_self_attention_map) * sd_cross_attention_maps2.max()
                #
                # if sd_cross_attention_maps2.max() >= threshold:
                #     binarized_sd_cross_attention_maps2 = torch.where(
                #         sd_cross_attention_maps2 > sd_cross_attention_maps2.mean() + 1 * sd_cross_attention_maps2.std(),
                #         1, 0)
                #     if torch.all(binarized_sd_cross_attention_maps2 == 0):
                #         binarized_sd_cross_attention_maps2 = torch.where(
                #             sd_cross_attention_maps2 > sd_cross_attention_maps2.mean(), 1, 0)
                #     masked_pixels_ids = torch.where(binarized_sd_cross_attention_maps2.flatten() == 1)[0]
                #     large_sd_self_attention_map = \
                #         torch.nn.functional.interpolate(sd_self_attention_maps[0][None, ...],
                #                                         crop_size, mode="bilinear")[0]
                #     avg_large_sd_self_attention_map = (sd_cross_attention_maps2.flatten()[masked_pixels_ids][..., None, None] * large_sd_self_attention_map[masked_pixels_ids]).mean(dim=0)
                #     avg_large_sd_self_attention_map = (avg_large_sd_self_attention_map - avg_large_sd_self_attention_map.min()) / (avg_large_sd_self_attention_map.max() - avg_large_sd_self_attention_map.min())
                #
                #     final_attention_map[y_start:y_end, x_start:x_end] += (avg_large_sd_self_attention_map * sd_cross_attention_maps2.max())
                #     aux_attention_map[y_start:y_end, x_start:x_end] += torch.ones_like(avg_large_sd_self_attention_map) * sd_cross_attention_maps2.max()

        return final_attention_map, aux_attention_map

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

        final_attention_map, aux_attention_map = self.get_patched_masks(image,
                                                                        self.config.crop_size,
                                                                        self.config.num_crops_per_side,
                                                                        self.config.crop_threshold
                                                                        )
        self.stable_diffusion.attention_maps = {}
        # if torch.all(final_attention_map == 0):
        #     final_max_attention_map /= aux_max_attention_map
        #     ones_indices = torch.where(aux_max_attention_map.flatten() > 0)[0]
        #     mean = final_max_attention_map.flatten()[ones_indices].mean()
        #     std = final_max_attention_map.flatten()[ones_indices].std()
        #     final_predicted_mask_0 = torch.where(final_max_attention_map > mean, 1, 0)
        #     final_predicted_mask_1 = torch.where(final_max_attention_map > mean + 1 * std, 1, 0)
        #     final_predicted_mask_2 = torch.where(final_max_attention_map > mean + 2 * std, 1, 0)
        # else:
        final_attention_map /= aux_attention_map
        # zero_indices = torch.where(final_attention_map.sum(dim=(1, 2)) == 0)[0]
        # final_attention_map[zero_indices] = final_max_attention_map[zero_indices]
        flattened_final_attention_map = final_attention_map.flatten(1, 2)
        minimum = flattened_final_attention_map.min(1, keepdim=True).values
        maximum = flattened_final_attention_map.max(1, keepdim=True).values
        flattened_final_attention_map = (flattened_final_attention_map - minimum) / (maximum - minimum + 1e-7)
        final_attention_map = flattened_final_attention_map.reshape(*final_attention_map.shape)
        final_mask = torch.argmax(final_attention_map, dim=0)

            # ones_indices = torch.where(aux_attention_map.flatten() > 0)[0]
            # mean = final_attention_map.flatten()[ones_indices].mean()
            # std = final_attention_map.flatten()[ones_indices].std()
            # final_predicted_mask_0 = torch.where(final_attention_map > mean, 1, 0)
            # final_predicted_mask_1 = torch.where(final_attention_map > mean + 1 * std, 1, 0)
            # final_predicted_mask_2 = torch.where(final_attention_map > mean + 2 * std, 1, 0)
        # final_predicted_mask_0 = final_predicted_mask_0.cpu()
        # final_predicted_mask_1 = final_predicted_mask_1.cpu()
        # final_predicted_mask_2 = final_predicted_mask_2.cpu()
        final_mask = final_mask.cpu()
        # final_mask = (final_mask / final_mask.max())
        # mask_grid_0 = torchvision.utils.make_grid(final_predicted_mask_0)
        # mask_grid_1 = torchvision.utils.make_grid(final_predicted_mask_1)
        # mask_grid_2 = torchvision.utils.make_grid(final_predicted_mask_2)
        mask_grid = torchvision.utils.make_grid(final_mask/final_mask.max())

        # masked_image_grid_0 = torchvision.utils.make_grid(
        #     image[0].cpu() * (
        #             1 - final_predicted_mask_0[None, ...]) + torch.stack(
        #         [final_predicted_mask_0 * 0, final_predicted_mask_0 * 0, final_predicted_mask_0], dim=0))
        # masked_image_grid_1 = torchvision.utils.make_grid(
        #     image[0].cpu() * (
        #             1 - final_predicted_mask_1[None, ...]) + torch.stack(
        #         [final_predicted_mask_1 * 0, final_predicted_mask_1 * 0, final_predicted_mask_1], dim=0))
        # masked_image_grid_2 = torchvision.utils.make_grid(
        #     image[0].cpu() * (
        #             1 - final_predicted_mask_2[None, ...]) + torch.stack(
        #         [final_predicted_mask_2 * 0, final_predicted_mask_2 * 0, final_predicted_mask_2], dim=0))

        masked_image_grid = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - final_mask[None, ...]) + torch.stack(
                [final_mask * 0, final_mask * 0, final_mask], dim=0))

        image_grid = torchvision.utils.make_grid(image)


        self.logger.experiment.add_image("test predicted mask", mask_grid, batch_idx)
        # self.logger.experiment.add_image("test predicted mask 0", mask_grid_0, batch_idx)
        # self.logger.experiment.add_image("test predicted mask 1", mask_grid_1, batch_idx)
        # self.logger.experiment.add_image("test predicted mask 2", mask_grid_2, batch_idx)
        self.logger.experiment.add_image("test masked image", masked_image_grid, batch_idx)
        # self.logger.experiment.add_image("test masked image 0", masked_image_grid_0, batch_idx)
        # self.logger.experiment.add_image("test masked image 1", masked_image_grid_1, batch_idx)
        # self.logger.experiment.add_image("test masked image 2", masked_image_grid_2, batch_idx)
        self.logger.experiment.add_image("test image", image_grid, batch_idx)

        if mask_provided:
            # if self.config.use_crf:
            #     crf_mask = torch.as_tensor(
            #         crf((image[0].permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy().copy(
            #             order='C'),
            #             torch.stack([final_predicted_mask_0, final_predicted_mask_0,
            #                          final_predicted_mask_0],
            #                         dim=2).type(torch.float).numpy()))[:, :, 0]
            #     crf_masked_image_grid1 = torchvision.utils.make_grid(
            #         image[0].cpu() * (1 - crf_mask[None, ...]).cpu() + crf_mask[None, ...])
            #     iou_0 = calculate_iou((crf_mask > 0).type(torch.uint8), (mask.cpu() > 0).type(torch.uint8))
            # else:
            for idx, part_name in enumerate(self.config.test_part_names):
                part_mask = torch.where(mask.cpu() == idx, 1, 0).type(torch.uint8)
                if torch.all(part_mask == 0):
                    continue
                iou = calculate_iou(torch.where(final_mask == idx, 1, 0).type(torch.uint8),
                                    part_mask)
                self.ious[part_name].append(iou.cpu())
                self.log(f"test {part_name} iou", iou, on_step=True, sync_dist=True)
                # iou_0 = calculate_iou((final_predicted_mask_0 > 0).type(torch.uint8),
                #                       (mask.cpu() > 0).type(torch.uint8))
                # iou_1 = calculate_iou((final_predicted_mask_1 > 0).type(torch.uint8),
                #                       (mask.cpu() > 0).type(torch.uint8))
                # iou_2 = calculate_iou((final_predicted_mask_2 > 0).type(torch.uint8),
                #                       (mask.cpu() > 0).type(torch.uint8))
            masks_grid = torchvision.utils.make_grid(mask[None, ...]/mask[None, ...].max())

        # if mask_provided:
            self.logger.experiment.add_image("test mask", masks_grid, batch_idx)
            # self.log("test iou 0", iou_0, on_step=True, sync_dist=True)
            # self.log("test iou 1", iou_1, on_step=True, sync_dist=True)
            # self.log("test iou 2", iou_2, on_step=True, sync_dist=True)
            # if self.config.use_crf:
            #     self.logger.experiment.add_image("test crf masked image1", crf_masked_image_grid1, batch_idx)

        return torch.tensor(0.)

    # def on_test_end(self) -> None:
    #     for part_name in self.ious:
    #         print(part_name, "    ", torch.stack(self.ious[part_name], 0).mean(0))

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            [
                {'params': self.translator.parameters(), 'lr': self.config.lr_t},
                # {'params': self.token_t, 'lr': self.config.lr_2},
                # {'params': self.token_u, 'lr': self.config.lr_1},

            ],
            lr=self.config.lr_1,
        )
        return {"optimizer": optimizer}

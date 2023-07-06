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
        translator_layers = [torch.nn.Linear(768, self.config.translator_hidden_layer_dim), torch.nn.ReLU()]
        # translator_layers = []
        for i in range(self.config.num_translator_hidden_layers):
            translator_layers.append(torch.nn.Linear(self.config.translator_hidden_layer_dim, self.config.translator_hidden_layer_dim))
            translator_layers.append(torch.nn.ReLU())
        translator_layers.append(torch.nn.Linear(self.config.translator_hidden_layer_dim, 768))
        self.translator = torch.nn.Sequential(
            *translator_layers
        )

    def on_train_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))
        self.uncond_embedding, self.text_embedding = self.stable_diffusion.get_text_embeds("object part", "object part")
        self.translator.train()

    def training_step(self, batch, batch_idx):
        src_images, mask1 = batch
        small_mask1 = torch.nn.functional.interpolate(mask1[None, ...], 64, mode="nearest")[0]
        mask1 = mask1[0]
        small_mask1 = small_mask1[0]
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
                indices = torch.where(small_mask1.flatten() == i)[0]
                if indices.shape[0] == 0:
                    continue
                self_attention_map = sd_self_attention_maps[0][torch.where(small_mask1.flatten() == i)[0]].sum(dim=0)
                loss3 = loss3 + torch.nn.functional.mse_loss(self_attention_map, torch.where(mask1 == i, 1., 0.))
                self_attention_maps.append(((self_attention_map - self_attention_map.min()) / (self_attention_map.max() - self_attention_map.min())).detach().cpu())
        if len(self_attention_maps) > 0:
            self_attention_maps = torch.stack(self_attention_maps, dim=0)
        loss2 = torch.tensor(0.)
        for layer_sd_cross_attention_maps2 in sd_cross_attention_maps2:
            loss2 = loss2 + torch.nn.functional.cross_entropy(layer_sd_cross_attention_maps2[None, ...], mask1[None, ...].type(torch.long))
        loss2 = loss2 / sd_cross_attention_maps2.shape[0]
        sd_cross_attention_maps2 = torch.unsqueeze(sd_cross_attention_maps2.mean(dim=0).softmax(dim=0), dim=1)
        sd_cross_attention_maps1 = torch.unsqueeze(sd_cross_attention_maps1.mean(dim=0).softmax(dim=0), dim=1)

        self.epoch_loss_2.append(loss2.detach().item())

        loss = loss2 + self.config.self_attention_loss_coef * loss3

        self.log("loss2", loss2.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss3", loss3.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss", loss.detach().cpu(), on_step=True, sync_dist=True)

        sd_cross_attention_maps2 = (sd_cross_attention_maps2 - sd_cross_attention_maps2.min()) / (
                sd_cross_attention_maps2.max() - sd_cross_attention_maps2.min())

        if sd_self_attention_maps is not None:
            self_attention_map_grid = torchvision.utils.make_grid(self_attention_maps.unsqueeze(dim=1))
            self.logger.experiment.add_image("train self attention map", self_attention_map_grid, self.counter)

        images_grid = torchvision.utils.make_grid(src_images.detach().cpu())
        sd_attention_maps_grid1 = torchvision.utils.make_grid(sd_cross_attention_maps1.detach().cpu())
        sd_attention_maps_grid2 = torchvision.utils.make_grid(sd_cross_attention_maps2.detach().cpu())
        mask1_grid = torchvision.utils.make_grid(mask1[None, ...].detach().cpu()/10)

        self.logger.experiment.add_image("train image", images_grid, self.counter)
        self.logger.experiment.add_image("train sd attention maps1", sd_attention_maps_grid1, self.counter)
        self.logger.experiment.add_image("train sd attention maps2", sd_attention_maps_grid2, self.counter)
        self.logger.experiment.add_image("train mask1", mask1_grid, self.counter)
        self.counter += 1

        return loss

    def on_train_epoch_end(self) -> None:
        mean_epoch_loss_2 = sum(self.epoch_loss_2) / len(self.epoch_loss_2)
        if mean_epoch_loss_2 < self.min_loss_2:
            self.min_loss_2 = mean_epoch_loss_2
            torch.save(self.translator.state_dict(),
                       os.path.join(self.config.train_checkpoint_dir, "translator.pth"))

    def on_train_end(self) -> None:
        torch.save(self.stable_diffusion.noise, os.path.join(self.config.train_checkpoint_dir, "noise.pth"))

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device, torch.device(f"cuda:{self.config.second_gpu_id}"))
        self.stable_diffusion.change_hooks(attention_layers_to_use=self.config.attention_layers_to_use)  # exclude self attention layer
        self.uncond_embedding, self.text_embedding = self.stable_diffusion.get_text_embeds("object part", "object part")
        self.translator.load_state_dict(torch.load(os.path.join(self.config.test_checkpoint_dir, "translator.pth")))
        self.translator.eval()

    def get_patched_masks(self, image, crop_size, num_crops_per_side, threshold):
        crops_coords = get_crops_coords(image.shape[2:], crop_size,
                                        num_crops_per_side)

        final_attention_map = torch.zeros(len(self.config.test_part_names), image.shape[2], image.shape[3])
        aux_attention_map = torch.zeros(len(self.config.test_part_names), image.shape[2], image.shape[3], dtype=torch.uint8) + 1e-7

        for i in range(num_crops_per_side):
            for j in range(num_crops_per_side):
                y_start, y_end, x_start, x_end = crops_coords[i * num_crops_per_side + j]
                cropped_image = image[:, :, y_start:y_end, x_start:x_end]
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
                # sd_cross_attention_maps2 : len(self.config.test_checkpoint_dir), 64, 64

                attention_maps = sd_cross_attention_maps2.flatten(1, 2).detach()  # len(self.config.test_checkpoint_dir) , 64x64
                max_values = attention_maps.max(dim=1).values  # len(self.config.test_checkpoint_dir)
                min_values = attention_maps.min(dim=1).values  # len(self.config.test_checkpoint_dir)
                passed_indices = torch.where(max_values >= threshold)[0]  #
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
                        final_attention_map[mask_id, y_start:y_end, x_start:x_end] += (
                                (avg_self_attention_map / coef) + (
                                    min_values[mask_id] - avg_self_attention_map_min / coef)).cpu()
                        # final_attention_map[mask_id, y_start:y_end, x_start:x_end] += (
                        #             avg_self_attention_map * max_values[mask_id])
                        aux_attention_map[mask_id, y_start:y_end, x_start:x_end] += (torch.ones_like(avg_self_attention_map, dtype=torch.uint8)).cpu()

        return final_attention_map, aux_attention_map

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask_provided = not torch.all(mask == 0)
        if mask_provided:
            mask = mask[0]

        final_attention_map, aux_attention_map = self.get_patched_masks(image,
                                                                        self.config.crop_size,
                                                                        self.config.num_crops_per_side,
                                                                        self.config.crop_threshold
                                                                        )
        self.stable_diffusion.attention_maps = {}
        final_attention_map /= aux_attention_map
        flattened_final_attention_map = final_attention_map.flatten(1, 2)
        minimum = flattened_final_attention_map.min(1, keepdim=True).values
        maximum = flattened_final_attention_map.max(1, keepdim=True).values
        final_mask = torch.where(flattened_final_attention_map > (maximum + minimum)/2, flattened_final_attention_map, 0).reshape(*final_attention_map.shape).argmax(0)

        final_mask = final_mask.cpu()

        mask_grid = torchvision.utils.make_grid(final_mask/final_mask.max())

        masked_image_grid = torchvision.utils.make_grid(
            image[0].cpu() * (
                    1 - final_mask[None, ...]) + torch.stack(
                [final_mask * 0, final_mask * 0, final_mask], dim=0))

        image_grid = torchvision.utils.make_grid(image)

        self.logger.experiment.add_image("test predicted mask", mask_grid, batch_idx)
        self.logger.experiment.add_image("test masked image", masked_image_grid, batch_idx)
        self.logger.experiment.add_image("test image", image_grid, batch_idx)

        if mask_provided:
            for idx, part_name in enumerate(self.config.test_part_names):
                part_mask = torch.where(mask.cpu() == idx, 1, 0).type(torch.uint8)
                if torch.all(part_mask == 0):
                    continue
                iou = calculate_iou(torch.where(final_mask == idx, 1, 0).type(torch.uint8),
                                    part_mask)
                self.ious[part_name].append(iou.cpu())
                self.log(f"test {part_name} iou", iou, on_step=True, sync_dist=True)
            masks_grid = torchvision.utils.make_grid(mask[None, ...]/mask[None, ...].max())
            self.logger.experiment.add_image("test mask", masks_grid, batch_idx)
        return torch.tensor(0.)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            [
                {'params': self.translator.parameters(), 'lr': self.config.lr_t},
            ],
            lr=self.config.lr_1,
        )
        return {"optimizer": optimizer}

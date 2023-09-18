import os

import pytorch_lightning as pl
import torch
import torchvision
from torch import optim
import torch.nn.functional as F

from src.config import Config
from src.stable_difusion import StableDiffusion
from src.utils import calculate_iou, get_crops_coords, get_square_cropping_coords
from src.pixel_classifier import PixelClassifier
import gc
from glob import glob


class CoSegmenterTrainer(pl.LightningModule):
    def __init__(self, config: Config, learning_rate=0.001):
        super().__init__()
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.max_val_iou = 0
        self.val_ious = []

        self.stable_diffusion = StableDiffusion(
            sd_version="2.1",
            partial_run=False,
            attention_layers_to_use=config.attention_layers_to_use,
        )
        if self.config.noise_dir is not None:
            noise = torch.load(self.config.noise_dir)
            self.stable_diffusion.noise = noise
            self.generate_noise = False
        else:
            self.generate_noise = True

        self.prepare_text_embeddings()
        del self.stable_diffusion.tokenizer
        del self.stable_diffusion.text_encoder
        torch.cuda.empty_cache()

        if self.config.objective_to_optimize == "translator":
            self.translator = torch.nn.Sequential(torch.nn.Linear(1024, 1024))
            self.translator[0].weight.data.zero_()
            self.translator[0].bias.data.zero_()
        elif self.config.objective_to_optimize == "text_embedding":
            if self.config.train:
                self.embeddings_to_optimize = []
                if self.config.trained_embeddings_dir is not None:
                    embeddings_paths = sorted(
                        glob(
                            os.path.join(
                                self.config.trained_embeddings_dir, "embedding*"
                            )
                        ),
                        key=lambda x: int(
                            x.split("/")[-1].split("_")[-1].split(".")[0]
                        ),
                    )
                    for embedding_path in embeddings_paths:
                        self.embeddings_to_optimize.append(torch.load(embedding_path))
                else:
                    for i in range(1, len(self.config.parts_to_return)):
                        embedding = self.text_embedding[:, i : i + 1].clone()
                        embedding.requires_grad_(True)
                        self.embeddings_to_optimize.append(embedding)

        self.checkpoint_dir = self.config.checkpoint_dir
        if self.config.first_stage_epoch < self.config.epochs:
            self.pixel_classifier = PixelClassifier(
                len(self.config.parts_to_return),
                640 + 320 + len(self.config.parts_to_return),
            )
        self.stage = "train_segmentation"
        if self.config.first_stage_epoch == 0:
            assert (
                self.config.trained_embeddings_dir is not None
            ), "trained text embeddings are not provided"
            self.stage = "refine_segmentation"
        if self.config.use_all_tokens_for_training:
            self.token_ids = list(range(77))
        else:
            self.token_ids = list(range(len(self.config.parts_to_return)))

    def prepare_text_embeddings(self):
        if self.config.text_prompt == "part_name":
            self.text_prompt = " ".join(
                ["part" for _ in range(len(self.config.parts_to_return[1:]))]
            )
        else:
            self.text_prompt = self.config.text_prompt
        (
            self.uncond_embedding,
            self.text_embedding,
        ) = self.stable_diffusion.get_text_embeds(self.text_prompt, "")

    def on_fit_start(self) -> None:
        self.checkpoint_dir = (
            f"{self.config.checkpoint_dir}_{self.logger.log_dir.split('/')[-1]}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.config.second_gpu_id is None:
            self.stable_diffusion.setup(self.device)
        else:
            self.stable_diffusion.setup(
                self.device, torch.device(f"cuda:{self.config.second_gpu_id}")
            )
        self.uncond_embedding, self.text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)

    def on_train_epoch_start(self) -> None:
        self.generate_noise = True
        if (
            self.stage == "train_segmentation"
            and self.current_epoch == self.config.first_stage_epoch
        ):
            self.stage == "refine_segmentation"
            self.config.masking = "pixel_classifier"
        # self.train_ious = []

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            optimizer.zero_grad()
        # text_embeddings_optimizer, pixel_classifier_optimizer = self.optimizers()
        # text_embeddings_optimizer.zero_grad()
        # pixel_classifier_optimizer.zero_grad()
        src_images, mask = batch
        if self.config.ce_weighting == "adaptive":
            num_pixels = torch.zeros(
                len(self.config.parts_to_return), dtype=torch.int64
            )
            values, counts = torch.unique(mask, return_counts=True)
            num_pixels[values.type(torch.int64).cpu()] = counts.type(torch.int64).cpu()
            num_pixels[0] = 0
            pixel_weights = torch.where(
                num_pixels > 0, num_pixels.sum() / num_pixels, 0
            )
            pixel_weights[0] = 1
        elif self.config.ce_weighting == "constant":
            pixel_weights = torch.ones(
                len(self.config.parts_to_return), dtype=torch.float
            )
        mask = mask[0]
        # t = torch.randint(low=10, high=160, size=(1,)).item()
        # _, text_embeddings = self.stable_diffusion.get_text_embeds("", "")
        if self.config.objective_to_optimize == "text_embedding":
            text_embedding = torch.cat(
                [
                    self.text_embedding[:, 0:1],
                    *list(
                        map(lambda x: x.to(self.device), self.embeddings_to_optimize)
                    ),
                    self.text_embedding[:, 1 + len(self.embeddings_to_optimize) :],
                ],
                dim=1,
            )
            # text_embedding = torch.cat([*list(map(lambda x:x.to(self.device), self.embeddings_to_optimize)), self.text_embedding[:, len(self.embeddings_to_optimize):]], dim=1)
            t_embedding = torch.cat([self.uncond_embedding, text_embedding])
        elif self.config.objective_to_optimize == "translator":
            t_embedding = torch.cat(
                [
                    self.uncond_embedding,
                    self.text_embedding + self.translator(self.text_embedding),
                ]
            )

        (
            sd_loss,
            _,
            sd_cross_attention_maps2,
            sd_self_attention_maps,
            unet_features,
        ) = self.stable_diffusion.train_step(
            t_embedding,
            src_images,
            guidance_scale=self.config.guidance_scale,
            t=torch.tensor(self.config.train_t),
            generate_new_noise=self.generate_noise,
            attention_output_size=self.config.train_mask_size,
            token_ids=self.token_ids,
            train=True,
            average_layers=True,
            apply_softmax=False,
        )
        loss = 0
        if self.config.sample_noise_on_epoch:
            self.generate_noise = False
        if self.stage == "train_segmentation":
            loss1 = F.cross_entropy(
                sd_cross_attention_maps2[None, ...],
                mask[None, ...].type(torch.long),
                weight=pixel_weights.to(self.stable_diffusion.device),
            )
            loss = loss1 + self.config.sd_loss_coef * sd_loss
            if self.config.self_attention_loss_coef > 0:
                sd_cross_attention_maps2 = sd_cross_attention_maps2.softmax(dim=0)
                loss2 = 0
                # if sd_self_attention_maps is not None:
                #     small_sd_cross_attention_maps2 = F.interpolate(sd_cross_attention_maps2[None, ...], 64, mode="bilinear")[0]
                #     self_attention_map = (sd_self_attention_maps * small_sd_cross_attention_maps2.flatten(1,2)[..., None, None]).sum(dim=1)
                #     one_shot_mask = torch.zeros(len(self.config.parts_to_return), mask.shape[0], mask.shape[1]).to(mask.device).scatter_(0, mask.unsqueeze(0).type(torch.int64), 1.)
                #     loss2 = F.mse_loss(self_attention_map, one_shot_mask)
                # self_atttention_maps = []
                small_sd_cross_attention_maps2 = F.interpolate(
                    sd_cross_attention_maps2[None, ...], 64, mode="bilinear"
                )[0]
                for i in range(len(self.config.parts_to_return)):
                    self_attention_map = (
                        sd_self_attention_maps
                        * small_sd_cross_attention_maps2[i].flatten()[..., None, None]
                    ).sum(dim=0)
                    loss2 = loss2 + F.mse_loss(
                        self_attention_map, torch.where(mask == i, 1.0, 0.0)
                    )
                sd_self_attention_maps = None
                small_sd_cross_attention_maps2 = None
                self_attention_map = None
                sd_cross_attention_maps2 = torch.unsqueeze(
                    sd_cross_attention_maps2, dim=1
                )

                loss += self.config.self_attention_loss_coef * loss2
                self.log("loss2", loss2.detach().cpu(), on_step=True, sync_dist=True)
            self.log("sd_loss", sd_loss.detach().cpu(), on_step=True, sync_dist=True)
            self.log("loss1", loss1.detach().cpu(), on_step=True, sync_dist=True)
            self.manual_backward(loss)
            optimizers[0].step()

        self.test_t_embedding = t_embedding
        if self.config.masking == "patched_masking":
            final_mask = self.get_patched_masks(
                src_images,
                self.config.train_mask_size,
            )
        elif self.config.masking == "zoomed_masking":
            final_mask = self.zoom_and_mask(
                src_images,
                batch_idx,
                self.config.train_mask_size,
            )
        elif self.config.masking == "pixel_classifier":
            unet_features = torch.cat(
                list(
                    map(
                        lambda x: F.interpolate(
                            x[None, ...], self.config.train_mask_size, mode="bilinear"
                        )[0],
                        unet_features,
                    )
                ),
                dim=0,
            )
            features = torch.cat([sd_cross_attention_maps2, unet_features], dim=0)
            # logits = self.pixel_classifier(
            #     features.reshape(
            #         -1, self.config.train_mask_size * self.config.train_mask_size
            #     ).permute(1, 0)
            # )
            logits = self.pixel_classifier(features[None, ...])
            # pixel_classification_loss = F.cross_entropy(
            #     logits, mask.flatten().type(torch.long)
            # )
            pixel_classification_loss = F.cross_entropy(
                logits, mask.type(torch.long)[None, ...]
            )
            loss += self.config.pixel_classifier_loss_coef * pixel_classification_loss
            # final_mask = (
            #     torch.softmax(logits.detach(), dim=1)
            #     .argmax(1)
            #     .reshape(self.config.train_mask_size, self.config.train_mask_size)
            #     .cpu()
            # )
            final_mask = logits[0].argmax(0).detach().cpu()
        else:
            final_mask = sd_cross_attention_maps2.argmax(0).cpu()
        if self.stage == "refine_segmentation":
            self.log(
                "p_c_loss",
                pixel_classification_loss.detach().cpu(),
                on_step=True,
                sync_dist=True,
            )
            self.manual_backward(loss)
            optimizers[1].step()
        # if self.current_epoch >= self.config.first_stage_epoch:
        sd_cross_attention_maps2 = None
        unet_features = None
        ious = []
        for idx, part_name in enumerate(self.config.parts_to_return):
            if self.config.dataset == "ade20k" and idx == 0:
                continue
            part_mask = torch.where(mask.cpu() == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"train {part_name} iou", iou, on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)
        # self.train_ious.append(mean_iou)
        self.log("train mean iou", mean_iou, on_step=True, sync_dist=True)
        self.log("loss", loss.detach().cpu(), on_step=True, sync_dist=True)

        if self.config.log_images:
            images_grid = torchvision.utils.make_grid(src_images.detach().cpu())
            sd_attention_maps_grid2 = torchvision.utils.make_grid(
                sd_cross_attention_maps2[:10].detach().cpu()
            )
            mask_grid = torchvision.utils.make_grid(
                mask[None, ...].detach().cpu() / mask[None, ...].detach().cpu().max()
            )

            self.logger.experiment.add_image("train image", images_grid, self.counter)
            self.logger.experiment.add_image(
                "train sd attention maps2", sd_attention_maps_grid2, self.counter
            )
            self.logger.experiment.add_image("train mask", mask_grid, self.counter)
            # self.logger.log_image(key="train image", images=src_images.detach().cpu(), step=self.counter)
            # self.logger.log_image(key="train sd attention maps2", images=sd_cross_attention_maps2.detach().cpu(), step=self.counter)
            # self.logger.log_image(key="train mask", images=mask[None, ...].detach().cpu(), step=self.counter)
            self.counter += 1
        # self.manual_backward(loss)
        # text_embeddings_optimizer.step()
        # pixel_classifier_optimizer.step()

    # def on_train_epoch_end(self):
    #     if self.config.val_data_ids is None:
    #         print("save weight on train end")
    #         epoch_mean_iou = sum(self.train_ious) / len(self.train_ious)
    #         if epoch_mean_iou >= self.max_train_iou:
    #             self.max_train_iou = epoch_mean_iou
    #             if self.config.objective_to_optimize == "text_embedding":
    #                 for i, embedding in enumerate(self.embeddings_to_optimize):
    #                     torch.save(
    #                         embedding,
    #                         os.path.join(self.checkpoint_dir, f"embedding_{i}.pth"),
    #                     )
    #                 # torch.save(self.token_0,
    #                 #         os.path.join(self.checkpoint_dir, "token_0.pth"))
    #             elif self.config.objective_to_optimize == "translator":
    #                 torch.save(
    #                     self.translator.state_dict(),
    #                     os.path.join(self.checkpoint_dir, "translator.pth"),
    #                 )
    #             # if self.config.sample_noise_on_epoch:
    #             #     torch.save(
    #             #         self.stable_diffusion.noise.cpu(),
    #             #         os.path.join(self.checkpoint_dir, "noise.pth"),
    #             #     )
    #         gc.collect()
    def get_simple_masks(self, image):
        with torch.no_grad():
            (
                _,
                _,
                sd_cross_attention_maps2,
                _,
                _,
            ) = self.stable_diffusion.train_step(
                self.test_t_embedding,
                image,
                guidance_scale=self.config.guidance_scale,
                t=torch.tensor(self.config.test_t),
                generate_new_noise=True,
                attention_output_size=self.config.test_mask_size,
                token_ids=self.token_ids,
                train=False,
            )
            final_mask = sd_cross_attention_maps2.argmax(0).cpu()
        return final_mask

    def get_pixel_classified_masks(self, image):
        with torch.no_grad():
            (
                _,
                _,
                sd_cross_attention_maps2,
                _,
                unet_features,
            ) = self.stable_diffusion.train_step(
                self.test_t_embedding,
                image,
                guidance_scale=self.config.guidance_scale,
                t=torch.tensor(self.config.test_t),
                generate_new_noise=True,
                attention_output_size=self.config.test_mask_size,
                token_ids=self.token_ids,
                train=False,
            )
            unet_features = torch.cat(
                list(
                    map(
                        lambda x: F.interpolate(
                            x[None, ...], self.config.test_mask_size, mode="bilinear"
                        )[0],
                        unet_features,
                    )
                ),
                dim=0,
            )
            features = torch.cat([sd_cross_attention_maps2, unet_features], dim=0)
            # logits = self.pixel_classifier(
            #     features.reshape(
            #         -1, self.config.test_mask_size * self.config.test_mask_size
            #     ).permute(1, 0)
            # )
            logits = self.pixel_classifier(features[None, ...])
        # prediction_mask = (
        #     torch.softmax(logits, dim=1)
        #     .argmax(1)
        #     .reshape(self.config.test_mask_size, self.config.test_mask_size)
        #     .cpu()
        # )
        prediction_mask = logits[0].argmax(0).cpu()
        return prediction_mask

    def get_patched_masks(self, image, output_size):
        crops_coords = get_crops_coords(
            image.shape[2:],
            self.config.patch_size,
            self.config.num_patchs_per_side,
        )

        final_attention_map = torch.zeros(
            len(self.config.parts_to_return),
            output_size,
            output_size,
        )

        aux_attention_map = (
            torch.zeros(
                len(self.config.parts_to_return),
                output_size,
                output_size,
                dtype=torch.uint8,
            )
            + 1e-7
        )

        ratio = 512 // output_size
        mask_patch_size = self.config.patch_size // ratio
        for crop_coord in crops_coords:
            y_start, y_end, x_start, x_end = crop_coord
            mask_y_start, mask_y_end, mask_x_start, mask_x_end = (
                y_start // ratio,
                y_end // ratio,
                x_start // ratio,
                x_end // ratio,
            )
            cropped_image = image[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                (
                    _,
                    _,
                    sd_cross_attention_maps2,
                    sd_self_attention_maps,
                    _,
                ) = self.stable_diffusion.train_step(
                    self.test_t_embedding,
                    cropped_image,
                    guidance_scale=self.config.guidance_scale,
                    t=torch.tensor(self.config.test_t),
                    generate_new_noise=True,
                    attention_output_size=64,
                    token_ids=self.token_ids,
                    train=False,
                )

                sd_cross_attention_maps2 = sd_cross_attention_maps2.flatten(
                    1, 2
                )  # len(self.config.checkpoint_dir) , 64x64

                max_values = sd_cross_attention_maps2.max(
                    dim=1
                ).values.cpu()  # len(self.config.checkpoint_dir)
                min_values = sd_cross_attention_maps2.min(
                    dim=1
                ).values.cpu()  # len(self.config.checkpoint_dir)
                passed_indices = torch.where(max_values >= self.config.patch_threshold)[
                    0
                ]  #
                if len(passed_indices) > 0:
                    sd_cross_attention_maps2 = sd_cross_attention_maps2[passed_indices]
                    sd_cross_attention_maps2[0] = torch.where(
                        sd_cross_attention_maps2[0]
                        > sd_cross_attention_maps2[0].mean(),
                        sd_cross_attention_maps2[0],
                        0,
                    )
                    for idx, mask_id in enumerate(passed_indices):
                        if not self.config.not_use_self_attention:
                            avg_self_attention_map = (
                                sd_cross_attention_maps2[idx][..., None, None]
                                * sd_self_attention_maps
                            ).sum(dim=0)
                            avg_self_attention_map = F.interpolate(
                                avg_self_attention_map[None, None, ...],
                                mask_patch_size,
                                mode="bilinear",
                            )[0, 0].cpu()

                            avg_self_attention_map_min = avg_self_attention_map.min()
                            avg_self_attention_map_max = avg_self_attention_map.max()
                            coef = (
                                avg_self_attention_map_max - avg_self_attention_map_min
                            ) / (max_values[mask_id] - min_values[mask_id])
                            if torch.isnan(coef) or coef == 0:
                                coef = 1e-7
                            final_attention_map[
                                mask_id,
                                mask_y_start:mask_y_end,
                                mask_x_start:mask_x_end,
                            ] += (avg_self_attention_map / coef) + (
                                min_values[mask_id] - avg_self_attention_map_min / coef
                            )
                            aux_attention_map[
                                mask_id,
                                mask_y_start:mask_y_end,
                                mask_x_start:mask_x_end,
                            ] += torch.ones_like(
                                avg_self_attention_map, dtype=torch.uint8
                            )
                        else:
                            final_attention_map[
                                mask_id,
                                mask_y_start:mask_y_end,
                                mask_x_start:mask_x_end,
                            ] += F.interpolate(
                                sd_cross_attention_maps2[idx].reshape(64, 64)[
                                    None, None, ...
                                ],
                                mask_patch_size,
                                mode="bilinear",
                            )[
                                0, 0
                            ]

        final_attention_map /= aux_attention_map
        final_mask = final_attention_map.argmax(0)
        return final_mask

    def zoom_and_mask(self, image, batch_idx, output_size):
        final_attention_map = torch.zeros(
            len(self.config.parts_to_return),
            output_size,
            output_size,
        )
        with torch.no_grad():
            (
                _,
                _,
                sd_cross_attention_maps2,
                sd_self_attention_maps,
                _,
            ) = self.stable_diffusion.train_step(
                self.test_t_embedding,
                image,
                guidance_scale=self.config.guidance_scale,
                t=torch.tensor(self.config.test_t),
                generate_new_noise=True,
                attention_output_size=64,
                token_ids=list(range(len(self.config.parts_to_return))),
                train=False,
            )
            self_attention_map = sd_self_attention_maps
            self_attention_map = F.interpolate(
                self_attention_map[None, ...],
                (output_size, output_size),
                mode="bilinear",
            )[0]
            attention_maps = sd_cross_attention_maps2.flatten(
                1, 2
            )  # len(self.config.checkpoint_dir) , 64x64
            sd_cross_attention_maps2 = None
            sd_self_attention_maps = None
            max_values = attention_maps.max(
                dim=1
            ).values  # len(self.config.checkpoint_dir)
            min_values = attention_maps.min(
                dim=1
            ).values  # len(self.config.checkpoint_dir)
            passed_indices = torch.where(max_values >= 0)[0]  #
            if len(passed_indices) > 0:
                passed_attention_maps = attention_maps[passed_indices]
                for idx, mask_id in enumerate(passed_indices):
                    if not self.config.not_use_self_attention:
                        avg_self_attention_map = (
                            passed_attention_maps[idx][..., None, None]
                            * self_attention_map
                        ).mean(dim=0)
                        avg_self_attention_map_min = avg_self_attention_map.min()
                        avg_self_attention_map_max = avg_self_attention_map.max()
                        coef = (
                            avg_self_attention_map_max - avg_self_attention_map_min
                        ) / (max_values[mask_id] - min_values[mask_id])
                        final_attention_map[mask_id] += (
                            (avg_self_attention_map / coef)
                            + (min_values[mask_id] - avg_self_attention_map_min / coef)
                        ).cpu()
                    else:
                        final_attention_map[mask_id] += F.interpolate(
                            passed_attention_maps[idx].reshape(64, 64)[None, None, ...],
                            (output_size, output_size),
                            mode="bilinear",
                        )[0, 0].cpu()
                    avg_self_attention_map = None

            final_mask = final_attention_map.argmax(0)

        if self.config.skip_zooming:
            return final_mask

        if torch.sum(torch.where(final_mask > 0, 1, 0)) == 0:
            x_start, x_end, y_start, y_end, patch_size = (
                0,
                image.shape[2],
                0,
                image.shape[2],
                image.shape[2],
            )
        else:
            x_start, x_end, y_start, y_end, patch_size = get_square_cropping_coords(
                torch.where(final_mask > 0, 1, 0),
                margin=self.config.crop_margin,
                min_square_size=self.config.min_square_size,
                original_size=image.shape[2],
            )
        ratio = 512 // output_size
        mask_y_start, mask_y_end, mask_x_start, mask_x_end = (
            y_start // ratio,
            y_end // ratio,
            x_start // ratio,
            x_end // ratio,
        )
        mask_patch_size = patch_size // ratio
        cropped_image = image[:, :, y_start:y_end, x_start:x_end]

        if self.config.log_images:
            final_mask_grid = torchvision.utils.make_grid(final_mask)
            image_grid = torchvision.utils.make_grid(cropped_image)
            masked_image = image[0].cpu() * (1 - final_mask[None, ...]) + torch.stack(
                [final_mask * 0, final_mask * 0, final_mask], dim=0
            )
            masked_image_grid = torchvision.utils.make_grid(masked_image)
            self.logger.experiment.add_image(
                "mask before zoom", final_mask_grid, batch_idx
            )
            self.logger.experiment.add_image(
                "masked image before zoom", masked_image_grid, batch_idx
            )
            self.logger.experiment.add_image("cropped image", image_grid, batch_idx)
        final_attention_map = torch.zeros(
            len(self.config.parts_to_return), mask_patch_size, mask_patch_size
        )
        with torch.no_grad():
            (
                _,
                _,
                sd_cross_attention_maps2,
                sd_self_attention_maps,
                _,
            ) = self.stable_diffusion.train_step(
                self.test_t_embedding,
                cropped_image,
                guidance_scale=self.config.guidance_scale,
                t=torch.tensor(self.config.test_t),
                generate_new_noise=True,
                attention_output_size=64,
                token_ids=list(range(len(self.config.parts_to_return))),
                train=False,
            )

            self_attention_map = F.interpolate(
                sd_self_attention_maps[None, ...], mask_patch_size, mode="bilinear"
            )[0]

            # self_attention_map : 64x64, 64, 64
            # sd_cross_attention_maps2 : len(self.config.checkpoint_dir), 64, 64

            attention_maps = sd_cross_attention_maps2.flatten(
                1, 2
            )  # len(self.config.checkpoint_dir) , 64x64
            max_values = attention_maps.max(
                dim=1
            ).values  # len(self.config.checkpoint_dir)
            min_values = attention_maps.min(
                dim=1
            ).values  # len(self.config.checkpoint_dir)
            passed_indices = torch.where(max_values >= self.config.patch_threshold)[
                0
            ]  #
            if len(passed_indices) > 0:
                passed_attention_maps = attention_maps[passed_indices]
                for idx, mask_id in enumerate(passed_indices):
                    if not self.config.not_use_self_attention:
                        avg_self_attention_map = (
                            passed_attention_maps[idx][..., None, None]
                            * self_attention_map
                        ).mean(dim=0)
                        avg_self_attention_map_min = avg_self_attention_map.min()
                        avg_self_attention_map_max = avg_self_attention_map.max()
                        coef = (
                            avg_self_attention_map_max - avg_self_attention_map_min
                        ) / (max_values[mask_id] - min_values[mask_id])
                        final_attention_map[mask_id] += (
                            (avg_self_attention_map / coef)
                            + (min_values[mask_id] - avg_self_attention_map_min / coef)
                        ).cpu()
                    else:
                        final_attention_map[mask_id] += F.interpolate(
                            passed_attention_maps[idx].reshape(64, 64)[None, None, ...],
                            patch_size,
                            mode="bilinear",
                        )[0, 0].cpu()
        crop_mask = final_attention_map.argmax(0)
        final_mask = torch.zeros(output_size, output_size)
        final_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = crop_mask

        return final_mask

    def on_validation_start(self):
        if self.config.objective_to_optimize == "text_embedding":
            text_embedding = torch.cat(
                [
                    self.text_embedding[:, 0:1],
                    *list(
                        map(
                            lambda x: x.to(self.device).detach(),
                            self.embeddings_to_optimize,
                        )
                    ),
                    self.text_embedding[:, 1 + len(self.embeddings_to_optimize) :],
                ],
                dim=1,
            )
            self.test_t_embedding = torch.cat([self.uncond_embedding, text_embedding])
        elif self.config.objective_to_optimize == "translator":
            self.test_t_embedding = torch.cat(
                [
                    self.uncond_embedding,
                    self.text_embedding + self.translator(self.text_embedding),
                ]
            )

    def on_validation_epoch_start(self):
        self.val_ious = []

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask[0]
        if self.config.masking == "patched_masking":
            final_mask = self.get_patched_masks(
                image,
                self.config.test_mask_size,
            )
        elif self.config.masking == "zoomed_masking":
            final_mask = self.zoom_and_mask(
                image,
                batch_idx,
                self.config.test_mask_size,
            )
        elif self.config.masking == "pixel_classifier":
            final_mask = self.get_pixel_classified_masks(image)
        else:
            final_mask = self.get_simple_masks(image)
        # final_mask = final_mask.cpu()
        # if self.config.log_images:
        if True:
            predicted_mask_grid = torchvision.utils.make_grid(
                final_mask / final_mask.max()
            )
            image_grid = torchvision.utils.make_grid(image)
            mask_grid = torchvision.utils.make_grid(
                mask[None, ...] / mask[None, ...].max()
            )

            self.logger.experiment.add_image("val mask", mask_grid, 0)
            self.logger.experiment.add_image(
                "val predicted mask", predicted_mask_grid, self.val_counter
            )
            self.logger.experiment.add_image("val image", image_grid, 0)
            self.val_counter += 1
        ious = []
        for idx, part_name in enumerate(self.config.parts_to_return):
            part_mask = torch.where(mask.cpu() == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            ).cpu()
            ious.append(iou)
            self.log(f"val {part_name} iou", iou, on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)
        self.val_ious.append(mean_iou)
        self.log("val mean iou", mean_iou, on_step=True, sync_dist=True)
        return torch.tensor(0.0)

    def on_validation_epoch_end(self):
        epoch_mean_iou = sum(self.val_ious) / len(self.val_ious)
        if epoch_mean_iou >= self.max_val_iou:
            self.max_val_iou = epoch_mean_iou
            if self.current_epoch < self.config.first_stage_epoch:
                if self.config.objective_to_optimize == "text_embedding":
                    for i, embedding in enumerate(self.embeddings_to_optimize):
                        torch.save(
                            embedding,
                            os.path.join(self.checkpoint_dir, f"embedding_{i}.pth"),
                        )
                    # torch.save(self.token_0,
                    #         os.path.join(self.checkpoint_dir, "token_0.pth"))
                elif self.config.objective_to_optimize == "translator":
                    torch.save(
                        self.translator.state_dict(),
                        os.path.join(self.checkpoint_dir, "translator.pth"),
                    )
            if self.config.masking == "pixel_classifier":
                torch.save(
                    self.pixel_classifier.state_dict(),
                    os.path.join(self.checkpoint_dir, "pixel_classifier.pth"),
                )
            # if self.config.sample_noise_on_epoch:
            #     torch.save(
            #         self.stable_diffusion.noise.cpu(),
            #         os.path.join(self.checkpoint_dir, "noise.pth"),
            #     )
        gc.collect()

    def on_test_start(self) -> None:
        if not self.config.train:
            if self.config.second_gpu_id is None:
                self.stable_diffusion.setup(self.device)
            else:
                self.stable_diffusion.setup(
                    self.device, torch.device(f"cuda:{self.config.second_gpu_id}")
                )
        #     uncond_embedding, text_embedding = self.stable_diffusion.get_text_embeds(
        #         self.text_prompt, ""
        #     )
        # else:
        uncond_embedding, text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)
        self.stable_diffusion.change_hooks(
            attention_layers_to_use=self.config.attention_layers_to_use
        )  # exclude self attention layer
        embeddings_to_optimize = []
        if self.config.objective_to_optimize == "text_embedding":
            for i in range(1, len(self.config.parts_to_return)):
                embedding = torch.load(
                    os.path.join(self.checkpoint_dir, f"embedding_{i-1}.pth")
                )
                embeddings_to_optimize.append(embedding)
            # token_0 = torch.load(os.path.join(self.checkpoint_dir, "token_0.pth"))
            text_embedding = torch.cat(
                [
                    text_embedding[:, 0:1],
                    *list(map(lambda x: x.to(self.device), embeddings_to_optimize)),
                    text_embedding[:, 1 + len(embeddings_to_optimize) :],
                ],
                dim=1,
            )
            # text_embedding = torch.cat(
            #     [token_0.to(self.stable_diffusion.device), token_t.to(self.stable_diffusion.device), text_embedding[:, 2:]], dim=1)
            self.test_t_embedding = torch.cat([uncond_embedding, text_embedding])
        elif self.config.objective_to_optimize == "translator":
            self.translator.load_state_dict(
                torch.load(os.path.join(self.checkpoint_dir, "translator.pth"))
            )
            self.test_t_embedding = torch.cat(
                [uncond_embedding, text_embedding + self.translator(text_embedding)]
            )
        if self.config.masking == "pixel_classifier":
            self.pixel_classifier.load_state_dict(
                torch.load(os.path.join(self.checkpoint_dir, "pixel_classifier.pth"))
            )
        # if self.config.sample_noise_on_epoch:
        #     noise = torch.load(os.path.join(self.checkpoint_dir, "noise.pth"))
        #     self.stable_diffusion.noise = noise.to(self.stable_diffusion.device)
        #     self.generate_noise = False
        # self.ious = {}
        # for part_name in self.config.parts_to_return:
        #     self.ious[part_name] = []

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask_provided = not torch.all(mask == 0)
        if mask_provided:
            mask = mask[0]
        if self.config.masking == "patched_masking":
            # final_whole_attention_map = self.get_patched_masks(
            #     image,
            #     512,
            #     1,
            #     self.config.patch_threshold,
            # )
            final_mask = self.get_patched_masks(
                image,
                self.config.test_mask_size,
            )
            # final_attention_map = 0.6 * final_whole_attention_map + 0.4 * final_patched_attention_map
            # final_mask = final_attention_map.argmax(0)
        elif self.config.masking == "zoomed_masking":
            final_mask = self.zoom_and_mask(
                image,
                batch_idx,
                self.config.test_mask_size,
            )
        elif self.config.masking == "pixel_classifier":
            final_mask = self.get_pixel_classified_masks(image)
        if self.config.log_images:
            predicted_mask_grid = torchvision.utils.make_grid(
                final_mask / final_mask.max()
            )
            masked_image = image[0].cpu() * (1 - final_mask[None, ...]) + torch.stack(
                [final_mask * 0, final_mask * 0, final_mask], dim=0
            )
            masked_image_grid = torchvision.utils.make_grid(masked_image)

            image_grid = torchvision.utils.make_grid(image)

            self.logger.experiment.add_image(
                "test predicted mask", predicted_mask_grid, batch_idx
            )
            self.logger.experiment.add_image(
                "test masked image", masked_image_grid, batch_idx
            )
            self.logger.experiment.add_image("test image", image_grid, batch_idx)
        if mask_provided:
            if self.config.log_images:
                mask_grid = torchvision.utils.make_grid(
                    mask[None, ...] / mask[None, ...].max()
                )
                self.logger.experiment.add_image("test mask", mask_grid, batch_idx)
                # self.logger.log_image(key="test mask", images=mask[None, ...], step=batch_idx)
            for idx, part_name in enumerate(self.config.parts_to_return):
                part_mask = torch.where(mask.cpu() == idx, 1, 0).type(torch.uint8)
                if torch.all(part_mask == 0):
                    continue
                iou = calculate_iou(
                    torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
                )
                # self.ious[part_name].append(iou.cpu())
                self.log(f"test {part_name} iou", iou, on_step=True, sync_dist=True)

        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        print("max val mean iou: ", self.max_val_iou)

    def configure_optimizers(self):
        if self.config.objective_to_optimize == "translator":
            params = self.translator.parameters()
        elif self.config.objective_to_optimize == "text_embedding":
            params = self.embeddings_to_optimize
        optimizers = []
        text_embeddings_optimizer = getattr(optim, self.config.optimizer)(
            [
                {"params": params},
                # {'params': [self.token_t], 'lr': self.config.lr},
            ],
            lr=self.config.lr,
        )
        optimizers.append(text_embeddings_optimizer)
        if self.config.first_stage_epoch < self.config.epochs:
            pixel_classifier_optimizer = getattr(optim, self.config.optimizer)(
                self.pixel_classifier.parameters(),
                lr=self.config.pixel_classifier_lr,
            )
            optimizers.append(pixel_classifier_optimizer)
        return optimizers

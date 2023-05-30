import pytorch_lightning as pl
from torch import optim
from src.config import Config
import torch
from src.stable_difusion import StableDiffusion
import math
import os
import torchvision


class CoSegmenterTrainer(pl.LightningModule):
    def __init__(self, config: Config, learning_rate=0.001):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.automatic_optimization = True
        self.stable_diffusion = StableDiffusion(sd_version='1.4', return_attentions=True)
        if self.config.train:
            uncond_embeddings, text_embeddings = self.stable_diffusion.get_text_embeds("part", "")
            self.text_embedding = text_embeddings.clone()
            self.text_embedding.requires_grad_(True)
            self.sd_text_embeddings = torch.cat([uncond_embeddings, self.text_embedding])
        else:
            uncond_embeddings, text_embeddings = self.stable_diffusion.get_text_embeds("part", "")
            self.text_embedding = torch.load(os.path.join(self.config.base_dir, "optimized_text_embedding.pth"))
            self.sd_text_embeddings = torch.cat([uncond_embeddings, self.text_embedding])

    def on_fit_start(self) -> None:
        self.stable_diffusion.setup(self.device)

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)

    def training_step(self, batch, batch_idx):
        src_images, emulated_attention_maps, _, _, _ = batch
        images_grid = torchvision.utils.make_grid(src_images)
        attn_grid = torchvision.utils.make_grid(emulated_attention_maps)
        self.logger.experiment.add_image("images", images_grid, batch_idx)
        self.logger.experiment.add_image("attention", attn_grid, batch_idx)
        loss, raw_attention_maps = self.stable_diffusion.train_step(torch.repeat_interleave(self.sd_text_embeddings, self.config.batch_size, 0), src_images, t=torch.tensor(8), back_propagate_loss=False)
        all_processed_attention_maps_dict = {}
        for key in raw_attention_maps:
            resized_attention_maps = []
            for raw_attention_map in raw_attention_maps[key].chunk(src_images.shape[0]):
                channel, img_embed_len, text_embed_len = raw_attention_map.chunk(2)[1].shape
                reshaped_attention_map = raw_attention_map.chunk(2)[1].softmax(dim=-1).reshape(
                                                                                            channel,
                                                                                            int(math.sqrt(img_embed_len)),
                                                                                            int(math.sqrt(img_embed_len)),
                                                                                            text_embed_len
                                                                                        )
                resized_attention_maps.append(
                    torch.nn.functional.interpolate(reshaped_attention_map.permute(0, 3, 1, 2), size=src_images.shape[-2:],
                                                    mode='bilinear').mean(dim=0))
            all_processed_attention_maps_dict[key] = torch.stack(resized_attention_maps, dim=0)

        all_processed_attention_maps = list(all_processed_attention_maps_dict.values())
        sd_attention_maps = torch.stack(all_processed_attention_maps, dim=0).mean(dim=0)

        loss = torch.nn.functional.mse_loss(sd_attention_maps[:, 1].to(self.device), emulated_attention_maps)
        self.log("mse_loss", loss, on_step=True, sync_dist=True)
        return loss

    def on_train_end(self) -> None:
        torch.save(self.text_embedding, os.path.join(self.config.base_dir, "optimized_text_embedding.pth"))

    def test_step(self, batch, batch_idx):
        target_images, emulated_attention_maps, y, x, target_images_original = batch
        with torch.no_grad():
            loss, raw_attention_maps = self.stable_diffusion.train_step(
                torch.repeat_interleave(self.sd_text_embeddings, self.config.batch_size, 0), target_images, t=torch.tensor(8), back_propagate_loss=False)
        all_processed_attention_maps_dict = {}
        for key in raw_attention_maps:
            resized_attention_maps = []
            for raw_attention_map in raw_attention_maps[key].chunk(target_images.shape[0]):
                channel, img_embed_len, text_embed_len = raw_attention_map.chunk(2)[1].shape
                reshaped_attention_map = raw_attention_map.chunk(2)[1].softmax(dim=-1).reshape(
                    channel,
                    int(math.sqrt(img_embed_len)),
                    int(math.sqrt(img_embed_len)),
                    text_embed_len
                )
                resized_attention_maps.append(
                    torch.nn.functional.interpolate(reshaped_attention_map.permute(0, 3, 1, 2),
                                                    size=target_images.shape[-2:],
                                                    mode='bilinear').mean(dim=0))
            all_processed_attention_maps_dict[key] = torch.stack(resized_attention_maps, dim=0)

        all_processed_attention_maps = list(all_processed_attention_maps_dict.values())
        sd_attention_maps = torch.stack(all_processed_attention_maps, dim=0).mean(dim=0)
        all_cropped_attention_maps = torch.zeros((self.config.batch_size, int(512/self.config.crop_ratio), int(512/self.config.crop_ratio)))
        for i in range(self.config.batch_size):
            flattened_attention_map = sd_attention_maps[i, 1].flatten()
            values, indices = torch.sort(flattened_attention_map, descending=True)
            all_cropped_attention_maps[i, y[i]:y[i]+512, x[i]:x[i]+512] = torch.where(sd_attention_maps[i, 1]>values[200], 1, 0)
        final_attention_map = all_cropped_attention_maps.mean(dim=0)
        images_grid = torchvision.utils.make_grid(target_images_original)
        attn_images_grid = torchvision.utils.make_grid(target_images_original.to(self.device)*(1-final_attention_map[None, None, ...]).to(self.device))
        attn_grid = torchvision.utils.make_grid(final_attention_map)
        self.logger.experiment.add_image("images", images_grid, 0)
        self.logger.experiment.add_image("attn_images", attn_images_grid, 0)
        self.logger.experiment.add_image("attentions", attn_grid, 0)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            [self.text_embedding],
            lr=self.config.lr,
        )
        return {"optimizer": optimizer}

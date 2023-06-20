from transformers import CLIPTextModel, CLIPTokenizer, logging

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy


class StableDiffusion(nn.Module):
    def __init__(self, sd_version='2.0', step_guidance=None, partial_run=False, attention_layers_to_use=[]):
        super().__init__()

        self.sd_version = sd_version
        self.step_guidance = step_guidance
        self.partial_run = partial_run
        print(f'[INFO] loading stable diffusion...')

        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
            # model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "1.4":
            model_key = "CompVis/stable-diffusion-v1-4"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")

        def freeze_params(params):
            for param in params:
                param.requires_grad = False

        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.parameters())
        freeze_params(self.unet.parameters())

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

        self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler")
        # self.scheduler = PNDMScheduler.from_config(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        if step_guidance is None:
            self.min_step = int(self.num_train_timesteps * 0.020)
            # self.min_step = int(self.num_train_timesteps * 0.5)
            # self.max_step = int(self.num_train_timesteps * 0.5)
            self.max_step = int(self.num_train_timesteps * 0.980)
        # print(f"with min={self.min_step} and max={self.max_step}")
        self.alphas = self.scheduler.alphas_cumprod  # for convenience
        self.device = None
        self.device1 = None
        print(f'[INFO] loaded stable diffusion!')

        self.attention_maps = {}
        self.noise = None
        if self.partial_run:
            del self.unet.up_blocks[3]
            del self.unet.conv_norm_out
            del self.unet.conv_act
            del self.unet.conv_out

        def create_nested_hook(n):
            def hook(module, input, output):
                self.attention_maps[n] = output[1]

            return hook

        self.handles = []
        for module in attention_layers_to_use:
            self.handles.append(eval("self.unet." + module).register_forward_hook(create_nested_hook(module)))

    def change_hooks(self, attention_layers_to_use):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.attention_maps = {}

        def create_nested_hook_with_detach(n):
            def hook(module, input, output):
                self.attention_maps[n] = output[1].detach()

            return hook

        for module in attention_layers_to_use:
            self.handles.append(
                eval("self.unet." + module).register_forward_hook(create_nested_hook_with_detach(module)))

    def setup(self, device, device1=None):
        self.device1 = device if device1 is None else device1
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.unet = self.unet.to(self.device1)
        self.alphas = self.alphas.to(device)
        self.device = device

    def get_text_embeds(self, prompt, negative_prompt, **kwargs):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt').to(self.device)

        with torch.set_grad_enabled(False):
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt').to(self.device)

        with torch.set_grad_enabled(False):
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]

        return uncond_embeddings, text_embeddings

    def get_attention_map(self, raw_attention_maps, output_size=256, token_id=2, average_layers=True):
        cross_attention_maps = {}
        self_attention_maps = {}
        resnets = {}
        for layer in raw_attention_maps:
            if layer.endswith("attn2"):  # cross attentions
                split_attention_maps = torch.stack(raw_attention_maps[layer].chunk(2), dim=0)
                _, channel, img_embed_len, text_embed_len = split_attention_maps.shape
                reshaped_split_attention_maps = split_attention_maps.softmax(dim=-1)[:, :, :, token_id].reshape(
                    2,  # because of chunk
                    channel,
                    int(math.sqrt(img_embed_len)),
                    int(math.sqrt(img_embed_len)),
                )

                resized_reshaped_split_attention_maps = torch.nn.functional.interpolate(reshaped_split_attention_maps,
                                                                                        size=output_size,
                                                                                        mode='bilinear').mean(dim=1)
                cross_attention_maps[layer] = resized_reshaped_split_attention_maps

            elif layer.endswith("attn1"):  # self attentions
                channel, img_embed_len, img_embed_len = raw_attention_maps[layer].shape
                split_attention_maps = raw_attention_maps[layer][:channel // 2]
                reshaped_split_attention_maps = split_attention_maps.softmax(dim=-1).reshape(
                    # 2,  # because of chunk
                    channel // 2,
                    int(math.sqrt(img_embed_len)),
                    int(math.sqrt(img_embed_len)),
                    img_embed_len
                ).permute(0, 3, 1, 2)
                resized_reshaped_split_attention_maps = torch.nn.functional.interpolate(reshaped_split_attention_maps,
                                                                                        size=output_size,
                                                                                        mode='bilinear')
                resized_reshaped_split_attention_maps = resized_reshaped_split_attention_maps.mean(dim=0)
                self_attention_maps[layer] = resized_reshaped_split_attention_maps
            elif layer.endswith("conv1") or layer.endswith("conv2"):  # resnets
                resnets[layer] = raw_attention_maps[layer].detach()
        sd_cross_attention_maps = [None, None]
        sd_self_attention_maps = None

        if len(cross_attention_maps.values()) > 0:
            if average_layers:
                sd_cross_attention_maps = torch.stack(list(cross_attention_maps.values()), dim=0).mean(dim=0).to(
                    self.device)
            else:
                sd_cross_attention_maps = torch.stack(list(cross_attention_maps.values()), dim=1).to(
                    self.device)
        if len(self_attention_maps.values()) > 0:
            sd_self_attention_maps = list(map(lambda x: x.to(self.device), list(self_attention_maps.values())))

        if len(resnets.values()) > 0:
            r = list(map(lambda x: x.to(self.device), list(resnets.values())))

        return sd_cross_attention_maps[0], sd_cross_attention_maps[1], sd_self_attention_maps

    def train_step(self, text_embeddings, input_image, guidance_scale=100, t=None, back_propagate_loss=True,
                   generate_new_noise=True, phase=0, attention_map=None, loss_coef=1, latents=None,
                   attention_output_size=256, token_id=2, average_layers=True):

        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')
        if not (input_image.shape[-2] == 512 and input_image.shape[-1] == 512):
            input_image = F.interpolate(input_image, (512, 512), mode='bilinear', align_corners=False)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if t is None:
            if self.step_guidance is not None:
                min_step, max_step = self.step_guidance[phase]
            else:
                min_step, max_step = self.min_step, self.max_step
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long)
        # encode image into latents with vae, requires grad!
        # _t = time.time()
        if latents is None:
            latents = self.encode_imgs(input_image)
        if attention_map is not None:
            latents = latents * attention_map.to(self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')
        t = t.to(self.device)
        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.set_grad_enabled(True):
            # add noise
            if generate_new_noise:
                noise = torch.randn_like(latents).to(self.device)
                self.noise = noise
            else:
                noise = self.noise
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred_ = self.unet(latent_model_input.to(self.device1), t.to(self.device1),
                                    encoder_hidden_states=text_embeddings.to(self.device1),
                                    partial_run=self.partial_run).sample.to(self.device)

        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')
        sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps = self.get_attention_map(
            self.attention_maps, output_size=attention_output_size, token_id=token_id, average_layers=average_layers)
        # if not self.partial_run:
        #     # perform guidance (high scale from paper!)
        #     noise_pred_uncond, noise_pred_text = noise_pred_.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        #
        #     # w(t), sigma_t^2
        #     # w = (1 - self.alphas[t])
        #     w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        #     grad = w * (noise_pred - noise)
        #     # clip grad for stable training?
        #     # grad = grad.clamp(-10, 10)
        #     grad = torch.nan_to_num(grad)
        #     loss = F.mse_loss(noise_pred, noise.float())
        #     # manually backward, since we omitted an item in grad and cannot simply autodiff.
        #     # _t = time.time()
        #     if back_propagate_loss:
        #         latents.backward(gradient=grad * loss_coef, retain_graph=True)
        #     # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')
        #
        #     return loss, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps
        return 0, sd_cross_attention_maps1, sd_cross_attention_maps2, sd_self_attention_maps

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        all_attention_maps = []

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input.to(self.device1), t.to(self.device1),
                                           encoder_hidden_states=text_embeddings.to(self.device1))['sample'].to(
                        self.device)

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                all_attention_maps.append(deepcopy(self.attention_maps))

        return latents, all_attention_maps

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents, all_attention_maps = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                                           num_inference_steps=num_inference_steps,
                                                           guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # # # Img to Numpy
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return imgs, all_attention_maps

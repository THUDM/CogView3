import os
import copy
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig

from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ...util import get_obj_from_str, default
from ...modules.diffusionmodules.discretizer import generate_roughly_equally_spaced_steps, sub_generate_roughly_equally_spaced_steps


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class LinearRelayDiffusionLoss(StandardDiffusionLoss):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        partial_num_steps=500,
        blurring_schedule='linear',
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__(
            sigma_sampler_config,
            type=type,
            offset_noise_level=offset_noise_level,
            batch2model_keys=batch2model_keys,
        )

        self.blurring_schedule = blurring_schedule
        self.partial_num_steps = partial_num_steps

    
    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        lr_input = batch["lr_input"]

        rand = torch.randint(0, self.partial_num_steps, (input.shape[0],))
        sigmas = self.sigma_sampler(input.shape[0], rand).to(input.dtype).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
            noise = noise.to(input.dtype)
        rand = append_dims(rand, input.ndim).to(input.dtype).to(input.device)
        if self.blurring_schedule == 'linear':
            blurred_input = input * (1 - rand / self.partial_num_steps) + lr_input * (rand / self.partial_num_steps)
        elif self.blurring_schedule == 'sigma':
            max_sigmas = self.sigma_sampler(input.shape[0], torch.ones(input.shape[0])*self.partial_num_steps).to(input.dtype).to(input.device)
            blurred_input = input * (1 - sigmas / max_sigmas) + lr_input * (sigmas / max_sigmas)
        elif self.blurring_schedule == 'exp':
            rand_blurring = (1 - torch.exp(-(torch.sin((rand+1) / self.partial_num_steps * torch.pi / 2)**4))) / (1 - torch.exp(-torch.ones_like(rand)))
            blurred_input = input * (1 - rand_blurring) + lr_input * rand_blurring
        else:
            raise NotImplementedError
        noised_input = blurred_input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

class ZeroSNRDiffusionLoss(StandardDiffusionLoss):

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.dtype).to(input.device)
        additional_model_inputs['idx'] = idx

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim)
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred
        return self.get_loss(model_output, input, w)
    
    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        

"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...modules.diffusionmodules.discretizer import generate_roughly_equally_spaced_steps
from ...util import append_dims, default, instantiate_from_config

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}

def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(w.device).to(w.dtype)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb

class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        cfg_cond_scale: Union[int, None] = None,
        cfg_cond_embed_dim: Union[int, None] = 256,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )

        # pd params
        self.cfg_cond_scale = cfg_cond_scale
        self.cfg_cond_embed_dim = cfg_cond_embed_dim
        
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]]).float()

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        if self.cfg_cond_scale is not None:
            batch_size = x.shape[0]
            scale_emb = guidance_scale_embedding(torch.ones(batch_size, device=x.device) * self.cfg_cond_scale, embedding_dim=self.cfg_cond_embed_dim, dtype=x.dtype)
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), scale_emb=scale_emb)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class DDIMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_noise=0.1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0):

        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        d = to_d(x, sigma, denoised)
        dt = append_dims(next_sigma * (1 - s_noise**2)**0.5 - sigma, x.ndim)

        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)

        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                self.s_noise,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x

    
def relay_to_d(x, sigma, denoised, image, step, total_step):
    blurring_d = (denoised - image) / total_step
    blurring_denoised = image + blurring_d * step
    d = (x - blurring_denoised) / append_dims(sigma, x.ndim)
    return d, blurring_d
    

class LinearRelayEDMSampler(EulerEDMSampler):
    def __init__(self, partial_num_steps=20, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.partial_num_steps = partial_num_steps

    def __call__(self, denoiser, image, randn, cond, uc=None, num_steps=None):
        randn_unit = randn.clone()
        randn, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            randn, cond, uc, num_steps
        )
        x = None

        for i in self.get_sigma_gen(num_sigmas):
            if i < self.num_steps - self.partial_num_steps:
                continue
            if x is None:
                x = image + randn_unit * append_dims(s_in * sigmas[i], len(randn_unit.shape))

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                step=i - self.num_steps + self.partial_num_steps,
                image=image,
                index=self.num_steps - i,
            )

        return x

    def euler_step(self, x, d, dt, blurring_d):
        return x + dt * d + blurring_d

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, step=None, image=None, index=None):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        beta_t = next_sigma / sigma_hat * index / self.partial_num_steps - (index - 1) / self.partial_num_steps
        x = x * append_dims(next_sigma / sigma_hat, x.ndim) + denoised * append_dims(1 - next_sigma / sigma_hat + beta_t, x.ndim) - image * append_dims(beta_t, x.ndim)
        return x
    

class ZeroSNRDDIMSampler(SingleStepDiffusionSampler):
    def __init__(
        self,
        do_cfg=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.do_cfg = do_cfg

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, indices = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device, return_idx=True
        )
        uc = default(uc, cond)

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, indices
    
    def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, i=None, idx=None):
        additional_model_inputs = {}
        if self.do_cfg:
            additional_model_inputs['idx'] = torch.cat([x.new_ones([x.shape[0]]) * idx] * 2)
        else:
            additional_model_inputs['idx'] = torch.cat([x.new_ones([x.shape[0]]) * idx])
        denoised = denoiser(*self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs)
        denoised = self.guider(denoised, alpha_cumprod_sqrt, step=i, num_steps=self.num_steps)
        return denoised

    def sampler_step(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, denoiser, x, cond, uc=None, i=None, idx=None, return_denoised=False):
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, i, idx).to(torch.float32)
        if i == self.num_steps - 1:
            if return_denoised:
                return denoised, denoised
            return denoised

        a_t = ((1-next_alpha_cumprod_sqrt**2)/(1-alpha_cumprod_sqrt**2))**0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        if return_denoised:
            return x, denoised
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, alpha_cumprod_sqrts, num_sigmas, cond, uc, indices = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrts[i],
                s_in * alpha_cumprod_sqrts[i + 1],
                denoiser,
                x,
                cond,
                uc,
                i=i,
                idx=indices[self.num_steps-i-1],
            )

        return x

import math
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import default, get_obj_from_str, instantiate_from_config


class SATDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "jpg")
        network_config = model_config.get("network_config", None)
        network_wrapper = model_config.get("network_wrapper", None)
        denoiser_config = model_config.get("denoiser_config", None)
        sampler_config = model_config.get("sampler_config", None)
        conditioner_config = model_config.get("conditioner_config", None)
        first_stage_config = model_config.get("first_stage_config", None)
        loss_fn_config = model_config.get("loss_fn_config", None)
        scale_factor = model_config.get("scale_factor", 1.0)
        disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        untrainable_prefixs = model_config.get("untrainable_prefixs", ["first_stage_model", "conditioner"])
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        lr_scale = model_config.get("lr_scale", None)
        use_pd = model_config.get("use_pd", False)  # progressive distillation

        self.log_keys = log_keys
        self.input_key = input_key
        self.untrainable_prefixs = untrainable_prefixs
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.use_pd = use_pd

        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config["params"]["dtype"] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))

        first_stage_model = instantiate_from_config(first_stage_config).eval()
        for param in first_stage_model.parameters():
            param.requires_grad = False
        self.first_stage_model = first_stage_model

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch, **kwargs):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        if self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            lr_z = self.encode_first_stage(lr_x)
            batch["lr_input"] = lr_z
        x = self.encode_first_stage(x)
        # batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        target_size=None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)

        if target_size is not None:
            denoiser = lambda input, sigma, c, **additional_model_inputs: self.denoiser(
                self.model, input, sigma, c, target_size=target_size, **additional_model_inputs
            )
        else:
            denoiser = lambda input, sigma, c, **additional_model_inputs: self.denoiser(
                self.model, input, sigma, c, **additional_model_inputs
            )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        if isinstance(samples, list):
            for i in range(len(samples)):
                samples[i] = samples[i].to(self.dtype)
        else:
            samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def sample_relay(
        self,
        image: torch.Tensor,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.dtype).to(self.device)
        denoiser = lambda input, sigma, c, **additional_model_inputs: self.denoiser(
            self.model, input, sigma, c, **additional_model_inputs
        )
        samples = self.sampler(denoiser, image, randn, cond, uc=uc)
        if isinstance(samples, list):
            for i in range(len(samples)):
                samples[i] = samples[i].to(self.dtype)
        else:
            samples = samples.to(self.dtype)
        return samples

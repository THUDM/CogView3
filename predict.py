# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path

from diffusers import CogView3PlusPipeline
import torch


MODEL_URL = "https://weights.replicate.delivery/default/THUDM/CogView3/model_cache.tar"
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.pipe = CogView3PlusPipeline.from_pretrained(
            MODEL_CACHE, torch_dtype=torch.bfloat16  # from THUDM/CogView3-Plus-3B
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[
                512,
                576,
                640,
                704,
                768,
                832,
                896,
                960,
                1024,
                1088,
                1152,
                1216,
                1280,
                1344,
                1408,
                1472,
                1536,
                1600,
                1664,
                1728,
                1792,
                1856,
                1920,
                1984,
                2048,
            ],
            default=1024,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[
                512,
                576,
                640,
                704,
                768,
                832,
                896,
                960,
                1024,
                1088,
                1152,
                1216,
                1280,
                1344,
                1408,
                1472,
                1536,
                1600,
                1664,
                1728,
                1792,
                1856,
                1920,
                1984,
                2048,
            ],
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

        torch.cuda.empty_cache()

        out_path = "/tmp/out.png"

        image.save(out_path)
        return Path(out_path)

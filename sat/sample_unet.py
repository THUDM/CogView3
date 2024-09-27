import os
import math
import argparse
from tqdm import tqdm
from typing import List, Union
from omegaconf import ListConfig
from PIL import Image

import torch
import torch.nn.functional as functional
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint

from diffusion import SATDiffusionEngine
from arguments import get_args


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
        elif key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(math.prod(N))
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(device, dtype=torch.half)
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        elif key == "cfg_scale":
            batch[key] = torch.tensor([value_dict["cfg_scale"]]).to(device).repeat(math.prod(N))
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def perform_save_locally(save_path, samples, grid, only_save_grid=False):
    os.makedirs(save_path, exist_ok=True)

    if not only_save_grid:
        for i, sample in enumerate(samples):
            sample = 255.0 * rearrange(sample.numpy(), "c h w -> h w c")
            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(save_path, f"{i:09}.png"))

    if grid is not None:
        grid = 255.0 * rearrange(grid.numpy(), "c h w -> h w c")
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(save_path, f"grid.png"))


def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = args.sampling_image_size
    input_sample_dirs = None
    if args.relay_model is True:
        sample_func = model.sample_relay
        H, W, C, F = image_size, image_size, 4, 8
        assert args.input_dir is not None
        input_sample_dirs = os.listdir(args.input_dir)
        input_sample_dirs_and_rank = sorted([(int(name.split("_")[0]), name) for name in input_sample_dirs])
        input_sample_dirs = [os.path.join(args.input_dir, name) for _, name in input_sample_dirs_and_rank]
    else:
        sample_func = model.sample
        latent_dim = args.sampling_latent_dim
        f = args.sampling_f
        H, W, C, F = image_size, image_size, latent_dim, f
    num_samples = [args.batch_size]
    force_uc_zero_embeddings = ["txt"]
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "original_size_as_tuple": (image_size, image_size),
                "target_size_as_tuple": (image_size, image_size),
                "orig_height": image_size,
                "orig_width": image_size,
                "target_height": image_size,
                "target_width": image_size,
                "crop_coords_top": 0,
                "crop_coords_left": 0,
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
            if args.relay_model is True:
                input_sample_dir = input_sample_dirs[cnt]
                images = []
                for i in range(args.batch_size):
                    filepath = os.path.join(input_sample_dir, f"{i:09}.png")
                    image = Image.open(filepath).convert("RGB")
                    image = TT.ToTensor()(image) * 2 - 1
                    images.append(image[None, ...])
                images = torch.cat(images, dim=0)
                images = functional.interpolate(images, scale_factor=2, mode="bilinear", align_corners=False)
                images = images.to(torch.float16).cuda()
                images = model.encode_first_stage(images)
                samples_z = sample_func(images, c, uc=uc, batch_size=args.batch_size, shape=(C, H // F, W // F))
            else:
                samples_z = sample_func(c, uc=uc, batch_size=args.batch_size, shape=(C, H // F, W // F))
            samples_x = model.decode_first_stage(samples_z).to(torch.float32)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            batch_size = samples.shape[0]
            assert (batch_size // args.grid_num_columns) * args.grid_num_columns == batch_size

            if args.batch_size == 1:
                grid = None
            else:
                grid = make_grid(samples, nrow=args.grid_num_columns)

            save_path = os.path.join(args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:20])
            perform_save_locally(save_path, samples, grid)


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    sampling_main(args, model_cls=SATDiffusionEngine)

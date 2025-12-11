import os
import math
import fire
import numpy as np
import torch

from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import (
    create_carvekit_interface,
    load_and_preprocess,
    instantiate_from_config,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_config(config, ckpt):
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    model = instantiate_from_config(config.model)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

def preprocess_image(models, img, with_carvekit=True):
    if with_carvekit:
        arr = load_and_preprocess(models["carvekit"], img)
        arr = arr.astype(np.float32) / 255.0
        return arr

    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C)

    if arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
        white = np.ones_like(arr)
        arr = alpha * arr + (1.0 - alpha) * white
        arr = arr[..., :3]

    return arr

def to_tensor(img_arr, h, w):
    x = transforms.ToTensor()(img_arr).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0
    x = transforms.functional.resize(x, (h, w))
    return x

@torch.no_grad()
def sample_model(
    input_tensor,
    model,
    sampler,
    h,
    w,
    ddim_steps,
    n_samples,
    scale,
    ddim_eta,
    elevation,
    azimuth,
    radius,
):
    device_ = input_tensor.device

    with model.ema_scope():
        # conditioning from input image
        c = model.get_learned_conditioning(input_tensor)
        c = c.tile(n_samples, 1, 1)

        # camera conditioning [elev, sin(az), cos(az), radius]
        cam = torch.tensor(
            [elevation, math.sin(azimuth), math.cos(azimuth), radius],
            device=device_,
            dtype=torch.float32,
        ).view(1, 1, 4).repeat(n_samples, 1, 1)

        # concat and project
        c = torch.cat([c, cam], dim=-1).float()
        c = model.cc_projection(c)

        # image-noisy conditioning (concat)
        img_latent = model.encode_first_stage(input_tensor).mode().detach()
        img_latent = img_latent.repeat(n_samples, 1, 1, 1)

        cond = {
            "c_crossattn": [c],
            "c_concat": [img_latent],
        }

        # unconditional conditioning for guidance
        if scale != 1.0:
            uc = {
                "c_crossattn": [torch.zeros_like(c, device=device_)],
                "c_concat": [
                    torch.zeros(n_samples, 4, h // 8, w // 8, device=device_)
                ],
            }
        else:
            uc = None

        # DDIM sampling
        shape = (4, h // 8, w // 8)
        samples_latent, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            x_T=None,
        )

        x = model.decode_first_stage(samples_latent)
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    return x.cpu()

def generate_views_for_image(
    img,
    models,
    elevation=0.0,
    azimuth=0.0,
    radius=0.0,
    use_carvekit=True,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    ddim_eta=1.0,
    h=256,
    w=256,
):
    img_arr = preprocess_image(models, img, with_carvekit=use_carvekit)
    input_tensor = to_tensor(img_arr, h, w)

    # diffusion sampling
    sampler = DDIMSampler(models["turncam"])
    x_samples = sample_model(
        input_tensor,
        models["turncam"],
        sampler,
        h,
        w,
        ddim_steps,
        n_samples,
        scale,
        ddim_eta,
        elevation,
        azimuth,
        radius,
    )

    outputs = []
    for sample in x_samples:
        sample = 255.0 * rearrange(sample.numpy(), "c h w -> h w c")
        img_from_sample = Image.fromarray(sample.astype(np.uint8))
        outputs.append(img_from_sample)

    return outputs

def multi_view_inference(
    ckpt: str = "./105000.ckpt",
    config_path: str = "configs/sd-objaverse-finetune-c_concat-256.yaml",
    input_img_path: str = "dummy.png",
    output_dir: str = "./outputs",
    radius: float = 0.0,
):
    config = OmegaConf.load(config_path)
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "turncam": load_model_from_config(config, ckpt),
        "carvekit": create_carvekit_interface(),
    }

    img = Image.open(input_img_path)
    base = os.path.splitext(os.path.basename(input_img_path))[0]

    viewpoints = [
        ("front", 0, 0),
        ("right", 0, 90),
        ("back", 0, 180),
        ("left", 0, 270),
        ("top", 60, 0),
        ("bottom", -60, 0),
    ]

    for name, elev_deg, az_deg in viewpoints:
        preds = generate_views_for_image(
            img=img,
            models=models,
            elevation=math.radians(elev_deg),
            azimuth=math.radians(az_deg),
            radius=radius,
        )

        if not preds:
            continue

        pred_image = preds[-1]
        out_path = os.path.join(output_dir, f"{base}_{name}.png")
        pred_image.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    fire.Fire(multi_view_inference)

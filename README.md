# Paper Review: Zero-1-to-3

Our review of **"Zero-1-to-3: Zero-shot One Image to 3D Object"** (Liu et al., ICCV 2023), written for the **MVA Geometric Deep Learning** final project.

This repository holds the slides and a small script we wrote to run the paper's pretrained model: given a **single image** of an object, generate **six views** (front, right, back, left, top, bottom) with the camera-conditioned diffusion model.

**Slides**: [presentation.pdf](presentation.pdf)

---

## Table of Contents

- [What We Implemented](#what-we-implemented)
- [Results](#results)
- [Setup](#setup)
- [How to Run](#how-to-run)

---

## What We Implemented

This is a short review, so we only implemented **inference** with the released checkpoint in [inference.py](inference.py). The input image is cleaned with CarveKit, then the camera-conditioned diffusion model is sampled with DDIM (50 steps, guidance scale 3.0). We run it for six target poses (front, right, back, left at 0° elevation, top at +60° and bottom at -60°) and save one image per view.

---

## Results

Generated views for a range of objects:

![Vehicles, table and watch](res1.png)

![Toy, statue, camera, bottle and face](res2.png)

![Cluttered scenes](res3.png)

---

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU recommended (a CPU fallback exists, but it is very slow)
- The [official Zero-1-to-3 code](https://github.com/cvlab-columbia/zero123) (provides the `ldm` package and the configs)
- The pretrained checkpoint `105000.ckpt` from the paper's release

### Installation

```bash
git clone https://github.com/cvlab-columbia/zero123
cd zero123
pip install -r requirements.txt
pip install fire carvekit-colab
```

Download `105000.ckpt` and place it next to the script. Copy [inference.py](inference.py) into the root of the `zero123/zero123` folder, so that `ldm` and `configs/` are importable.

---

## How to Run

```bash
python inference.py --input_img_path my_object.png
```

Options (all optional):

| Argument | Default | Meaning |
|---|---|---|
| `--ckpt` | `./105000.ckpt` | Path to the checkpoint |
| `--config_path` | `configs/sd-objaverse-finetune-c_concat-256.yaml` | Model config |
| `--input_img_path` | `dummy.png` | Input image |
| `--output_dir` | `./outputs` | Where the views are saved |
| `--radius` | `0.0` | Change in camera distance |

The six views are saved as `outputs/<name>_front.png`, `..._right.png`, `..._back.png`, `..._left.png`, `..._top.png`, `..._bottom.png`.

---


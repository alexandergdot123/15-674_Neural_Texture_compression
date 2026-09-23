# 15-674 Neural Texture Compression

Compares S3TC (DXT1) texture compression against a neural texture
representation (multi-resolution feature grid + small MLP) on a set of
sample textures, evaluating both PSNR and storage size.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Pillow
- Matplotlib
- tqdm

Install dependencies:

```bash
pip install torch numpy pillow matplotlib tqdm
```

## Input textures

The script expects the following image files in the working directory:

- `gradient.png`
- `clouds.png`
- `bricks.png`
- `dirty_wall_texture.png`
- `flowers_texture.png`
- `lamp_panel.png`


- If you wish to run this on your own texture, declare an image as a variable using Image.open("my_texture.png").convert("RGB")
- Then convert it to a texture using texture_np = np.asarray(my_image, dtype=np.float32) / 255.0

- Finally, run run_experiment(name, texture_np, device, quantize) with quantize optionally turned on (turning it on will run both the quantized and non-quantized models)
## Running

```bash
python assignment_1_main.py
```

The script will, for each texture:

1. Compress it with a from-scratch DXT1 (S3TC) implementation and evaluate PSNR.
2. Train three neural texture models (`Small`, `Medium`, `Large` ) to reconstruct the texture from UV coordinates.
3. Quantize each trained model's feature grids to 8-bit and re-evaluate PSNR.
4. Print a size/PSNR comparison table to stdout.
5. Save a side-by-side visual comparison (`<texture_name>_comparison.png`)
   showing the original, S3TC reconstruction, and each neural model's
   reconstruction (before and after quantization).

Training runs on CPU by default (`get_device()` in the script; CUDA/MPS are
disabled). Each texture trains three models
for 2000 steps each.

## Output

- Console output: a table of KB size and PSNR for S3TC and each of the
  three neural model configs, with and without quantization.
- `texture1_comparison.png` through `texture6_comparison.png`: visual
  comparison grids saved to the working directory.



# 15-674 Gaussian Splatting

Implements a differentiable Gaussian splatting renderer from scratch, first fitting
2D Gaussians to single images (with adaptive densification), then reconstructing a
small 3D scene from posed views and evaluating it on held-out cameras.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Pillow
- Matplotlib

Install dependencies:

```bash
pip install torch torchvision numpy pillow matplotlib
```

For a recent NVIDIA GPU (e.g. RTX 40 series), install a CUDA build of PyTorch that
supports your card, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Input data

The script expects the following in the working directory:

- `coffee.png`
- `astronaut.png`
- `cat.png`
- `spheres/` (unzipped `spheres.zip`), containing `cameras.json`, `train/`, and `val/`

```
a2/
├── main.py
├── coffee.png
├── astronaut.png
├── cat.png
└── spheres/
    ├── cameras.json
    ├── train/
    └── val/
```

- If you wish to fit your own 2D image, call `fit("my_image.png", N)` for a plain fit with N Gaussians, or `fit("my_image.png", N, use_densify=True)` for a densified fit that starts at N/4 and grows toward N. It returns the final PSNR and saves `compare_my_image_<N>.png`.

## Running

```bash
python main.py
```

Each part can be turned on or off with the flags near the bottom of `main.py`:

```python
RUN_P4 = True
RUN_P5 = True
RUN_P7 = True
RUN_P8 = True
RUN_P9 = True
```

The script will:

1. (P4) Fit each image with a 1024-Gaussian budget, once plain and once with densification (prune, clone, split), and compare PSNR.
2. (P5) Fit each image at N = 256, 1024, and 4096 Gaussians and plot PSNR against N.
3. (P7) Fit 4000 3D Gaussians to the 44 training views for 1500 iterations and report mean training PSNR.
4. (P8) Fit the same scene with 3D densification, starting from 1000 Gaussians and growing toward 4000, and compare training and held-out PSNR against P7.
5. (P9) Evaluate the densified model on all 12 held-out views and render a 36-frame orbit around the scene.

P8 and P9 read the saved P7 and P8 results from `results/`, so once P7 has run, the later parts can be rerun on their own with the earlier flags set to `False`.

Training uses CUDA if available, then MPS, then CPU (`get_device()` in the script). Every run uses a fixed random seed, so results are reproducible. With a GPU, each 2D fit takes well under a minute, and each 3D fit takes a few minutes.

## Output

- Console output: per-part PSNR results, Gaussian counts, and densification gradient statistics.
- `compare_<image>_<N>.png` and `compare_<image>_<N>_dense.png`: target, render, and per-pixel error for each 2D fit, saved to the working directory.
- `results/`:
  - `p4_results.json`, `p5_results.json`, `p7_results.json`, `p8_results.json`, `p9_results.json`: all reported numbers.
  - `psnr_vs_n.png`: PSNR against Gaussian count for the three images.
  - `p7_train_views.png`: ground truth vs. render for three training views.
  - `p7_val_views.png`, `p8_val_views.png`: the same held-out views rendered by the plain and densified models.
  - `p9_all_val_views.png`: ground truth vs. render for all 12 held-out views.
  - `p9_orbit.gif`, `p9_orbit_00.png` through `p9_orbit_27.png`: orbit around the reconstructed scene.
  - `p7_model.pt`, `p8_model.pt`: trained 3D Gaussian parameters.

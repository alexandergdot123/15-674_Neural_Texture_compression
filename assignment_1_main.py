import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
def get_device():
    # if torch.cuda.is_available():
    #     return "cuda"
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"

def lerp(arr_a, arr_b, t):
    return arr_a * (1 - t) + arr_b * t

def get_texel(image, x, y): #image has to be a 3d numpy array, y, x, and rgb from outer to inner
    x_base = math.floor(x * len(image[0]))
    y_base = math.floor(y * len(image))
    fx = x * len(image[0]) - x_base
    fy = y * len(image) - y_base
    x_low = lerp(image[y_base][x_base], image[y_base][min(x_base + 1, len(image[0]) - 1)], fx)
    x_high = lerp(image[min(y_base + 1, len(image) - 1)][x_base], image[min(y_base + 1, len(image) - 1)][min(x_base + 1, len(image[0]) - 1)], fx)
    return lerp(x_low, x_high, fy)

def project_fraction(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    d = b - a
    denom = np.dot(d, d)
    if denom == 0.0:
        return 0.0
    return np.dot(c - a, d) / denom

def dxt1_compress(image): #assuming colors are from 0 to 1
    compressed_image = []
    for i in range(len(image) // 4):
        row_of_blocks = []
        for j in range((len(image[0])//4)):
            block_data = []
            smallest_color = [100000, 100000, 100000]
            largest_color = [-10000, -10000, -1000]
            for y in range(4):
                for x in range(4):
                    for z in range(3):
                        if image[i * 4 + y][j * 4 + x][z] > largest_color[z]:
                            largest_color[z] = image[i * 4 + y][j * 4 + x][z]
                        if image[i * 4 + y][j * 4 + x][z] < smallest_color[z]:
                            smallest_color[z] = image[i * 4 + y][j * 4 + x][z]
            block_data.append([round(smallest_color[0]* 31), round(smallest_color[1]* 63), round(smallest_color[2]* 31)])
            block_data.append([round(largest_color[0]* 31), round(largest_color[1]* 63), round(largest_color[2]* 31)])
            color_quantizer = []
            for y in range(4):
                color_quantizer_row = []
                for x in range(4):
                    projected_distance = project_fraction(smallest_color, largest_color, image[i * 4 + y][j * 4 + x])
                    projected_distance = min(max(projected_distance, 0.0), 1.0)
                    projected_distance *= 3
                    projected_distance = round(projected_distance)
                    color_quantizer_row.append(projected_distance)
                color_quantizer.append(color_quantizer_row)
            block_data.append(color_quantizer)
            row_of_blocks.append(block_data)
        compressed_image.append(row_of_blocks)
    return compressed_image

def decompress_dxt1_to_sample_single_pixel(compressed_image, x, y):
    block_x = x // 4
    block_y = y // 4
    block_data = compressed_image[block_y][block_x]
    color_quantizer = block_data[2]
    local_x = x % 4
    local_y = y % 4
    projected_distance = color_quantizer[local_y][local_x] / 3.0
    levels = np.array([31.0, 63.0, 31.0])
    c0 = np.array(block_data[0], dtype=float) / levels
    c1 = np.array(block_data[1], dtype=float) / levels
    return lerp(c0, c1, projected_distance)

def get_texel_compressed_dxt1(compressed_image, x, y):
    height = len(compressed_image) * 4
    width = len(compressed_image[0]) * 4
    x0 = min(math.floor(x), width - 1)
    y0 = min(math.floor(y), height - 1)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    a = decompress_dxt1_to_sample_single_pixel(compressed_image, x0, y0)
    b = decompress_dxt1_to_sample_single_pixel(compressed_image, x1, y0)
    c = decompress_dxt1_to_sample_single_pixel(compressed_image, x0, y1)
    d = decompress_dxt1_to_sample_single_pixel(compressed_image, x1, y1)
    fx = x - math.floor(x)
    fy = y - math.floor(y)
    x_low = lerp(a, b, fx)
    x_high = lerp(c, d, fx)
    return lerp(x_low, x_high, fy)

class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        self.resolutions = resolutions
        self.feat_dim = feat_dim
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(1, feat_dim, R, R) * 0.01)
            for R in resolutions
        ])
        self.out_dim = feat_dim * len(resolutions)

    def forward(self, uv):
        N = uv.shape[0]
        grid_coords = uv * 2.0 - 1.0
        grid_coords = grid_coords.view(1, N, 1, 2)
        feats = []
        for grid in self.grids:
            sampled = F.grid_sample(
                grid, grid_coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled = sampled.squeeze(0).squeeze(-1).t()
            feats.append(sampled)
        return torch.cat(feats, dim=-1)


class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class NeuralTexture(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feat_dim)
        self.mlp  = ColorMLP(self.grid.out_dim)
    def forward(self, uv):
        return self.mlp(self.grid(uv))


class NeuralTextureCompressor:
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2,
                 lr=1e-2, batch_size=16384, steps=2000, device=None):
        self.resolutions = resolutions
        self.feat_dim = feat_dim
        self.lr = lr
        self.batch_size = batch_size
        self.steps = steps
        self.device = device or get_device()
        self.model = None

    def compress(self, texture):
        image = torch.as_tensor(texture, dtype=torch.float32, device=self.device)
        H, W, _ = image.shape

        ys, xs = torch.meshgrid(
            (torch.arange(H, device=self.device) + 0.5) / H,
            (torch.arange(W, device=self.device) + 0.5) / W,
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        target = image.reshape(-1, 3)
        N = coords.shape[0]

        self.model = NeuralTexture().to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for step in range(self.steps):
            idx = torch.randint(0, N, (self.batch_size,), device=self.device)
            pred = self.model(coords[idx])
            loss = F.mse_loss(pred, target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        return self.model

    @torch.no_grad()
    def sample(self, u, v):
        uv = torch.tensor([[u, v]], dtype=torch.float32, device=self.device)
        return self.model(uv).squeeze(0).cpu().numpy()


MODEL_CONFIGS = {
    "Small":  dict(resolutions=(64,),            feat_dim=2),
    "Medium": dict(resolutions=(16, 32, 64),      feat_dim=2),
    "Large":  dict(resolutions=(16, 32, 64, 128), feat_dim=4),
}

def model_size_kb(model):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / 1024.0

def s3tc_size_kb(H, W):
    n_blocks = (H // 4) * (W // 4)
    return n_blocks * 8 / 1024.0

def raw_size_kb(H, W):
    return H * W * 3 / 1024.0

@torch.no_grad()
def full_image_eval(model, coords, target, H, W, chunk=65536):
    preds = [model(coords[i:i+chunk]) for i in range(0, coords.shape[0], chunk)]
    pred = torch.cat(preds, dim=0)
    mse = F.mse_loss(pred, target).item()
    psnr = -10 * math.log10(mse) if mse > 0 else float("inf")
    recon = pred.clamp(0, 1).cpu().numpy().reshape(H, W, 3)
    return psnr, recon

def s3tc_eval(image_np, compressed):
    H, W, _ = image_np.shape
    recon = np.zeros_like(image_np)
    for y in range(H):
        for x in range(W):
            recon[y, x] = decompress_dxt1_to_sample_single_pixel(compressed, x, y)
    mse = np.mean((recon - image_np) ** 2)
    psnr = -10 * math.log10(mse) if mse > 0 else float("inf")
    return psnr, recon

def quantize_uint8(x):
    lo = x.min().item()
    hi = x.max().item()
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    q = torch.round((x - lo) / scale).clamp(0, 255)
    x_hat = lo + q * scale
    return q, lo, scale, x_hat

def quantize_model(model):
    for grid in model.grid.grids:
        q, lo, scale, x_hat = quantize_uint8(grid.data)
        grid.data.copy_(x_hat)

def quantized_size_kb(model):
    grid_params = sum(g.numel() for g in model.grid.grids)
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    return (grid_params * 1 + mlp_params * 4) / 1024.0


def run_experiment(name, image_np, device, quantize=False):
    H, W, _ = image_np.shape
    image = torch.as_tensor(image_np, dtype=torch.float32, device=device)

    ys, xs = torch.meshgrid(
        (torch.arange(H, device=device) + 0.5) / H,
        (torch.arange(W, device=device) + 0.5) / W,
        indexing="ij",
    )
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    target = image.reshape(-1, 3)
    N = coords.shape[0]

    compressed = dxt1_compress(image_np)
    s3tc_psnr_val, s3tc_recon = s3tc_eval(image_np, compressed)
    s3tc_kb = s3tc_size_kb(H, W)
    raw_kb = raw_size_kb(H, W)

    results = {}
    for cfg_name, cfg in MODEL_CONFIGS.items():
        model = NeuralTexture(**cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        pbar = tqdm(range(2000), desc=f"{name} / {cfg_name}", leave=False)
        curve = []
        pbar = tqdm(range(2000), desc=f"{name} / {cfg_name}", leave=False)
        for step in pbar:
            idx = torch.randint(0, N, (16384,), device=device)
            pred = model(coords[idx])
            loss = F.mse_loss(pred, target[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 10 == 0 or step == 1999:
                curve.append((step, -10 * math.log10(loss.item())))
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}")
        psnr, recon = full_image_eval(model, coords, target, H, W)
        results[cfg_name] = dict(psnr=psnr, kb=model_size_kb(model), recon=recon, curve=curve)
        if quantize:
            quantize_model(model)
            psnr_q, recon_q = full_image_eval(model, coords, target, H, W)
            results[cfg_name]["psnr_q"] = psnr_q
            results[cfg_name]["kb_q"] = quantized_size_kb(model)
            results[cfg_name]["recon_q"] = recon_q

    print(f"\n=== {name} ({H}x{W}) ===")
    print(f"Raw:   {raw_kb:8.1f} KB")
    print(f"S3TC:  {s3tc_kb:8.1f} KB   PSNR {s3tc_psnr_val:6.2f} dB   ratio {raw_kb/s3tc_kb:5.2f}:1")
    for cfg_name in MODEL_CONFIGS:
        r = results[cfg_name]
        print(f"{cfg_name:6s}: {r['kb']:8.1f} KB   PSNR {r['psnr']:6.2f} dB   ratio {raw_kb/r['kb']:5.2f}:1")
        if quantize:
            print(f"{'':6s}  quantized: {r['kb_q']:8.1f} KB   PSNR {r['psnr_q']:6.2f} dB   ratio {raw_kb/r['kb_q']:5.2f}:1")

    n_rows = 2 if quantize else 1
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 5)

    axes[0, 0].imshow(image_np); axes[0, 0].set_title("Original")
    axes[0, 1].imshow(s3tc_recon); axes[0, 1].set_title(f"S3TC\n{s3tc_psnr_val:.1f} dB, {s3tc_kb:.1f} KB")
    for ax, cfg_name in zip(axes[0, 2:], MODEL_CONFIGS):
        r = results[cfg_name]
        ax.imshow(r["recon"]); ax.set_title(f"{cfg_name}\n{r['psnr']:.1f} dB, {r['kb']:.1f} KB")

    if quantize:
        axes[1, 0].imshow(image_np); axes[1, 0].set_title("Original")
        axes[1, 1].imshow(s3tc_recon); axes[1, 1].set_title(f"S3TC\n{s3tc_psnr_val:.1f} dB, {s3tc_kb:.1f} KB")
        for ax, cfg_name in zip(axes[1, 2:], MODEL_CONFIGS):
            r = results[cfg_name]
            ax.imshow(r["recon_q"])
            ax.set_title(f"{cfg_name} (quantized)\n{r['psnr_q']:.1f} dB, {r['kb_q']:.1f} KB")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(s3tc=(s3tc_psnr_val, s3tc_kb), raw_kb=raw_kb, **results)

def plot_training_curves(all_results, fname):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4.5))
    for ax, (name, res) in zip(axes, all_results.items()):
        for cfg in MODEL_CONFIGS:
            steps, psnrs = zip(*res[cfg]["curve"])
            ax.plot(steps, psnrs, label=cfg, linewidth=1.2)
        ax.axhline(res["s3tc"][0], ls="--", c="gray", label="S3TC")
        ax.set_title(name)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Minibatch PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_size_vs_quality(all_results, fname):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4.5))
    for ax, (name, res) in zip(axes, all_results.items()):
        cfgs = list(MODEL_CONFIGS)
        kb = [res[c]["kb"] for c in cfgs]
        ps = [res[c]["psnr"] for c in cfgs]
        kbq = [res[c]["kb_q"] for c in cfgs]
        psq = [res[c]["psnr_q"] for c in cfgs]
        ax.plot(kb, ps, "o-", label="Neural fp32")
        ax.plot(kbq, psq, "s--", label="Neural 8-bit grid")
        for c, x, y in zip(cfgs, kb, ps):
            ax.annotate(c, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        s3_psnr, s3_kb = res["s3tc"]
        ax.plot(s3_kb, s3_psnr, "k*", markersize=14, label="S3TC")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("Size (KB, log scale)")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    
import matplotlib.pyplot as plt
from PIL import Image

torch.manual_seed(0)
device = get_device()

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

provided = {"gradient": load("gradient.png"), "clouds": load("clouds.png"), "bricks": load("bricks.png")}
results_provided = {n: run_experiment(n, im, device, quantize=True) for n, im in provided.items()}
plot_training_curves(results_provided, "p6_training_curves.png")
plot_size_vs_quality(results_provided, "p6_p7_size_vs_quality.png")

own = {"dirty_wall": load("dirty_wall_texture.png"), "flowers": load("flowers_texture.png"),
       "lamp_panel": load("lamp_panel.png")}
results_own = {n: run_experiment(n, im, device, quantize=True) for n, im in own.items()}
plot_training_curves(results_own, "p8_training_curves.png")
plot_size_vs_quality(results_own, "p8_size_vs_quality.png")
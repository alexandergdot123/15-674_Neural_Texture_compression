# import torch
# import numpy as np, torch
# from PIL import Image
# import json, random
# from types import SimpleNamespace
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# densify_every  = 200
# grad_threshold = 2e-4
# size_threshold = 0.02
# split_scale    = 1.6
# prune_opacity  = 0.005

# def get_device():
#     if torch.cuda.is_available():
#         return "cuda"
#     if torch.backends.mps.is_available():
#         return "mps"
#     return "cpu"

# dev = get_device()

# def covariance_2d(scale, theta):
#     c = torch.cos(theta)
#     s = torch.sin(theta)
#     my_matrix = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
#     my_matrix = my_matrix * scale[:, None, :]
#     return my_matrix @ my_matrix.transpose(-1, -2)

# def gaussian_weight(xy, mu, Sigma):
#     inverse = torch.linalg.inv(Sigma)
#     a, b, c = inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]
#     dx = xy[:, 0:1] - mu[None, :, 0]
#     dy = xy[:, 1:2] - mu[None, :, 1]
#     my_matrix = a * dx**2 + 2 * b * dx * dy + c * dy**2
#     return torch.exp(-0.5 * my_matrix)


# def pixel_grid(H, W, device=None):
#     ys, xs = torch.meshgrid(torch.arange(H, device=device),
#                             torch.arange(W, device=device), indexing='ij')
#     return torch.stack([xs, ys], -1).reshape(-1, 2).float()

# # def render(mu, Sigma, color, opacity, order, H, W):
# #     xy = pixel_grid(H, W, device=mu.device)
# #     w  = gaussian_weight(xy, mu, Sigma)
# #     alpha = opacity[None, :] * w
# #     alpha = alpha[:, order]
# #     T = torch.cumprod(1 - alpha, dim=1)
# #     T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)
# #     C = (T * alpha) @ color[order]
# #     return C.reshape(H, W, 3)

# from torch.utils.checkpoint import checkpoint

# def render_pixels(xy, mu, Sigma, color, opacity, order):
#     w  = gaussian_weight(xy, mu, Sigma)
#     alpha = opacity[None, :] * w
#     alpha = alpha[:, order]
#     T = torch.cumprod(1 - alpha, dim=1)
#     T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)
#     return (T * alpha) @ color[order]                             # (chunk of P, 3)

# def render(mu, Sigma, color, opacity, order, H, W, chunks=8):
#     xy = pixel_grid(H, W, device=mu.device)
#     C = torch.cat([checkpoint(render_pixels, xy_c, mu, Sigma, color, opacity, order, use_reentrant=False)
#                    for xy_c in xy.chunk(chunks)])                 # each chunk is rendered separately
#     return C.reshape(H, W, 3)

# def densify(gaussians, grad_mag, budget, W):
#     mu, log_s, theta, color, op_raw = [t.detach() for t in gaussians]
#     keep = op_raw.sigmoid() >= prune_opacity
#     mu, log_s, theta, color, op_raw, grad_mag = [t[keep] for t in (mu, log_s, theta, color, op_raw, grad_mag)]
#     dense = grad_mag > grad_threshold
#     room = max(budget - mu.shape[0], 0)
#     if dense.sum() > room:
#         idx = torch.where(dense)[0]
#         top = idx[grad_mag[idx].topk(room).indices]
#         dense = torch.zeros_like(dense)
#         dense[top] = True
#     max_scale = log_s.exp().max(dim=1).values
#     clone = dense & (max_scale <= size_threshold * W)
#     split = dense & (max_scale >  size_threshold * W)
#     children = []
#     for i in range(2):
#         z = torch.randn_like(mu[split]) * log_s[split].exp()
#         c, s = torch.cos(theta[split]), torch.sin(theta[split])
#         offset = torch.stack([c * z[:, 0] - s * z[:, 1],
#                               s * z[:, 0] + c * z[:, 1]], -1)
#         children.append([mu[split] + offset, log_s[split] - torch.log(torch.tensor(split_scale)),
#                 theta[split], color[split], op_raw[split]])
#     child1, child2 = children[0], children[1]

#     old    = [t[~split] for t in (mu, log_s, theta, color, op_raw)]
#     clones = [t[clone]  for t in (mu, log_s, theta, color, op_raw)]

#     return tuple(torch.cat([o, k, c1, c2]).requires_grad_()
#                  for o, k, c1, c2 in zip(old, clones, child1, child2))





# def quaternion_to_rotation(q):
#     q = q / q.norm(dim=-1, keepdim=True)
#     w, x, y, z = q.unbind(-1)
#     R = torch.stack([torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
#         torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
#         torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),], -2)
#     return R

# def covariance_3d(scale, quat):
#     my_matrix = quaternion_to_rotation(quat)
#     my_matrix = my_matrix * scale[:, None, :]
#     return (my_matrix @ my_matrix.transpose(-1, -2))

# def project_gaussian(mu3, Sigma3, R_wc, t, K):
#     mu_cam = mu3 @ R_wc.T + t
#     x, y, z = mu_cam.unbind(-1)
#     fx = K[0, 0]
#     fy = K[1, 1]
#     cx = K[0, 2]
#     cy = K[1, 2]
#     mu2 = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)

#     zero = torch.zeros_like(z)
#     J = torch.stack([torch.stack([fx / z, zero,   -fx * x / z**2], -1),
#         torch.stack([zero,   fy / z, -fy * y / z**2], -1)], -2)

#     Scam = R_wc @ Sigma3 @ R_wc.T
#     Sig2 = J @ Scam @ J.transpose(-1, -2)
#     return mu2, Sig2, z

# def load_cameras(split):
#     with open(f"spheres/cameras.json") as f:
#         meta = json.load(f)
#     K = torch.tensor(meta["K"], dtype=torch.float32, device=dev)
#     cams = []
#     for frame in meta["frames"]:
#         if not frame["file"].startswith(split + "/"):
#             continue
#         img_np = np.asarray(Image.open(f"spheres/{frame['file']}").convert("RGB"), dtype=np.float32) / 255
#         cams.append(SimpleNamespace(
#             R=torch.tensor(frame["R_wc"], dtype=torch.float32, device=dev),
#             t=torch.tensor(frame["t"],    dtype=torch.float32, device=dev),
#             K=K,
#             image=torch.from_numpy(img_np).to(dev),
#             H=meta["height"], W=meta["width"]))
#     return cams

# def render_view(cam, mu3, log_s, quat, color, op_raw):
#     Sig3 = covariance_3d(log_s.exp(), quat)
#     mu2, Sig2, depth = project_gaussian(mu3, Sig3, cam.R, cam.t, cam.K)
#     order = torch.argsort(depth, descending=False)
#     return render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, cam.H, cam.W)

# def mean_psnr(cams, mu3, log_s, quat, color, op_raw):
#     total = 0.0
#     with torch.no_grad():
#         for cam in cams:
#             img = render_view(cam, mu3, log_s, quat, color, op_raw)
#             total += (-10 * torch.log10(((img - cam.image) ** 2).mean())).item()
#     total /= len(cams)
#     return total

# def save_views(cams, idxs, tag, mu3, log_s, quat, color, op_raw):
#     fig, ax = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)))
#     with torch.no_grad():
#         for row, i in enumerate(idxs):
#             img = render_view(cams[i], mu3, log_s, quat, color, op_raw).clamp(0, 1).cpu().numpy()
#             ax[row, 0].imshow(cams[i].image.cpu().numpy())
#             ax[row, 0].set_title(f"view {i} ground truth")
#             ax[row, 1].imshow(img)
#             ax[row, 1].set_title(f"view {i} render")
#             for a in ax[row]: a.axis("off")
#     plt.tight_layout()
#     plt.savefig(f"{tag}.png", dpi=150)
#     plt.close(fig)


# train_cameras = load_cameras("train")
# val_cameras   = load_cameras("val")
# print(len(train_cameras), len(val_cameras))

# torch.manual_seed(0)
# N = 4000
# mu3    = ((torch.rand(N, 3, device=dev) * 2 - 1) * 1.5).requires_grad_()
# log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev)).requires_grad_()
# quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0; quat.requires_grad_()
# color  = torch.zeros(N, 3, device=dev, requires_grad=True)
# op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)
# opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)

# iters = 1500
# for step in range(iters):
#     cam  = random.choice(train_cameras)
#     img  = render_view(cam, mu3, log_s, quat, color, op_raw)
#     loss = ((img - cam.image) ** 2).mean()
#     opt.zero_grad(); loss.backward(); opt.step()
#     if step % 100 == 0:
#         print(step, loss.item(), (-10 * torch.log10(loss)).item())

# print("mean train PSNR:", mean_psnr(train_cameras, mu3, log_s, quat, color, op_raw))
# save_views(train_cameras, [0, len(train_cameras) // 3, 2 * len(train_cameras) // 3], "p7_train_views",
#            mu3, log_s, quat, color, op_raw)

















# def fit(path, budget, use_densify=False, steps=2000):
#     torch.manual_seed(0)                                 # same init for every run
#     img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
#     target = torch.from_numpy(img_np).to(dev)
#     H, W = target.shape[:2]
#     N = budget // 4 if use_densify else budget

#     mu     = (torch.rand(N, 2, device=dev) * torch.tensor([W, H], device=dev)).requires_grad_()
#     log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2, device=dev)).requires_grad_()
#     theta  = torch.zeros(N, device=dev, requires_grad=True)
#     color  = torch.zeros(N, 3, device=dev, requires_grad=True)
#     op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)

#     opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
#     grad_accum = torch.zeros(N, device=dev)
#     steps_since = 0

#     for step in range(steps):
#         depth_order = torch.arange(mu.shape[0], device=dev)
#         Sigma = covariance_2d(log_s.exp(), theta)
#         img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order, H, W)
#         loss  = ((img - target) ** 2).mean()
#         opt.zero_grad(); loss.backward()
#         grad_accum += mu.grad.norm(dim=1)
#         steps_since += 1
#         opt.step()

#         if use_densify and (step + 1) % densify_every == 0 and step < int(0.75 * steps):
#             g = grad_accum / steps_since
#             mu, log_s, theta, color, op_raw = densify((mu, log_s, theta, color, op_raw), g, budget, W)
#             opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
#             grad_accum = torch.zeros(mu.shape[0], device=dev)
#             steps_since = 0

#     # final PSNR, measured the same way for every run (after the last update)
#     with torch.no_grad():
#         Sigma = covariance_2d(log_s.exp(), theta)
#         img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(),
#                        torch.arange(mu.shape[0], device=dev), H, W)
#         psnr  = (-10 * torch.log10(((img - target) ** 2).mean())).item()

#     out = img.clamp(0, 1).cpu().numpy()
#     err = np.abs(out - img_np).mean(axis=2)
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#     ax[0].imshow(img_np);            ax[0].set_title("target")
#     ax[1].imshow(out);               ax[1].set_title(f"render (N={mu.shape[0]}, {psnr:.2f} dB)")
#     ax[2].imshow(err, cmap="magma"); ax[2].set_title("abs error")
#     for a in ax: a.axis("off")
#     plt.tight_layout()
#     name = path.rsplit(".", 1)[0]
#     plt.savefig(f"compare_{name}_{budget}{'_dense' if use_densify else ''}.png", dpi=150)
#     plt.close(fig)

#     return psnr


# # images = ["coffee", "astronaut", "cat"]
# # counts = [256, 1024, 4096]
# # results = {}

# # for name in images:
# #     results[name] = []
# #     for N in counts:
# #         psnr = fit(f"{name}.png", N)
# #         results[name].append(psnr)
# #         print(f"{name:10s} N={N:5d}  size={N * 9:6d}  PSNR={psnr:.2f} dB")
# #         torch.cuda.empty_cache()

# # plt.figure(figsize=(6, 4))
# # for name in images:
# #     plt.plot(counts, results[name], marker="o", label=name)
# # plt.xscale("log", base=2)
# # plt.xticks(counts, [str(n) for n in counts])
# # plt.xlabel("number of Gaussians N")
# # plt.ylabel("PSNR (dB)")
# # plt.title("Quality vs. count")
# # plt.grid(True, alpha=0.3)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("psnr_vs_n.png", dpi=150)
# # with open("p5_results.json", "w") as f:
# #     json.dump({"counts": counts, "psnr": results}, f, indent=2)

# def densify_3d(gaussians, grad_mag, budget):
#     mu, log_s, theta, color, op_raw = [t.detach() for t in gaussians]
#     keep = op_raw.sigmoid() >= prune_opacity
#     keep_idx = torch.where(keep)[0]
#     mu, log_s, theta, color, op_raw, grad_mag = [t[keep] for t in (mu, log_s, theta, color, op_raw, grad_mag)]
#     dense = grad_mag > grad_threshold
#     room = max(budget - mu.shape[0], 0)
#     if dense.sum() > room:
#         idx = torch.where(dense)[0]
#         top = idx[grad_mag[idx].topk(room).indices]
#         dense = torch.zeros_like(dense)
#         dense[top] = True
#     max_scale = log_s.exp().max(dim=1).values
#     clone = dense & (max_scale <= size_threshold)
#     split = dense & (max_scale >  size_threshold)
#     children = []
#     for i in range(2):
#         z = torch.randn_like(mu[split]) * log_s[split].exp()
#         offset = (quaternion_to_rotation(theta[split]) @ z[:, :, None])
#         offset = offset[:, :, 0]
#         children.append([mu[split] + offset, log_s[split] - torch.log(torch.tensor(split_scale)),
#                 theta[split], color[split], op_raw[split]])
#     child1, child2 = children[0], children[1]
#     old    = [t[~split] for t in (mu, log_s, theta, color, op_raw)]
#     clones = [t[clone]  for t in (mu, log_s, theta, color, op_raw)]
#     return tuple(torch.cat([o, k, c1, c2]).requires_grad_() for o, k, c1, c2 in zip(old, clones, child1, child2)), keep_idx[~split]

# def rebuild(old_opt, old_params, new_params, survivors, lr=1e-2):
#     new_opt = torch.optim.Adam(new_params, lr=lr)
#     for p_old, p_new in zip(old_params, new_params):
#         state = old_opt.state.get(p_old)
#         if not state:
#             continue
#         exp_avg    = torch.zeros_like(p_new)
#         exp_avg_sq = torch.zeros_like(p_new)
#         exp_avg[:len(survivors)]    = state["exp_avg"][survivors]
#         exp_avg_sq[:len(survivors)] = state["exp_avg_sq"][survivors]
#         new_opt.state[p_new] = {"step": state["step"].clone(), "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
#     return new_opt


# torch.manual_seed(0)
# budget = 4000                                             # same as P7
# N = budget // 4                                           # start small, grow toward the budget
# mu3    = ((torch.rand(N, 3, device=dev) * 2 - 1) * 1.5).requires_grad_()
# log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev)).requires_grad_()
# quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0; quat.requires_grad_()
# color  = torch.zeros(N, 3, device=dev, requires_grad=True)
# op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)
# opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)
# grad_accum = torch.zeros(N, device=dev)
# steps_since = 0

# for step in range(iters):
#     cam  = random.choice(train_cameras)
#     img  = render_view(cam, mu3, log_s, quat, color, op_raw)
#     loss = ((img - cam.image) ** 2).mean()
#     opt.zero_grad(); loss.backward()
#     grad_accum += mu3.grad.norm(dim=1)                    # accumulates across the random cameras
#     steps_since += 1
#     opt.step()

#     if (step + 1) % densify_every == 0 and step < int(0.75 * iters):
#         g = grad_accum / steps_since
#         print("grad quantiles (50/90/99%):", torch.quantile(g, torch.tensor([0.5, 0.9, 0.99], device=dev)).tolist())
#         old_params = [mu3, log_s, quat, color, op_raw]
#         (mu3, log_s, quat, color, op_raw), survivors = densify_3d(tuple(old_params), g, budget)
#         opt = rebuild(opt, old_params, [mu3, log_s, quat, color, op_raw], survivors)
#         grad_accum = torch.zeros(mu3.shape[0], device=dev)
#         steps_since = 0

#     if step % 100 == 0:
#         print(step, mu3.shape[0], loss.item(), (-10 * torch.log10(loss)).item())

# p8_count = mu3.shape[0]
# p8_train = mean_psnr(train_cameras, mu3, log_s, quat, color, op_raw)
# p8_val   = mean_psnr(val_cameras,   mu3, log_s, quat, color, op_raw)
# save_views(val_cameras, [0], "p8_val_view", mu3, log_s, quat, color, op_raw)

# print(f"plain     N={p7_count}  train={p7_train:.2f}  val={p7_val:.2f}")
# print(f"densified N={p8_count}  train={p8_train:.2f}  val={p8_val:.2f}")

import torch
import numpy as np, torch
from PIL import Image
import json, random
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import os

densify_every  = 200
grad_threshold = 2e-4
size_threshold = 0.02
split_scale    = 1.6
prune_opacity  = 0.005

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

dev = get_device()

def covariance_2d(scale, theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    my_matrix = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    my_matrix = my_matrix * scale[:, None, :]
    return my_matrix @ my_matrix.transpose(-1, -2)

def gaussian_weight(xy, mu, Sigma):
    inverse = torch.linalg.inv(Sigma)
    a, b, c = inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]
    dx = xy[:, 0:1] - mu[None, :, 0]
    dy = xy[:, 1:2] - mu[None, :, 1]
    my_matrix = a * dx**2 + 2 * b * dx * dy + c * dy**2
    return torch.exp(-0.5 * my_matrix)


def pixel_grid(H, W, device=None):
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    return torch.stack([xs, ys], -1).reshape(-1, 2).float()

def render_pixels(xy, mu, Sigma, color, opacity, order):
    w  = gaussian_weight(xy, mu, Sigma)
    alpha = opacity[None, :] * w
    alpha = alpha[:, order]
    T = torch.cumprod(1 - alpha, dim=1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)
    return (T * alpha) @ color[order]                             # (chunk of P, 3)

def render(mu, Sigma, color, opacity, order, H, W, chunks=8):
    xy = pixel_grid(H, W, device=mu.device)
    C = torch.cat([checkpoint(render_pixels, xy_c, mu, Sigma, color, opacity, order, use_reentrant=False)
                   for xy_c in xy.chunk(chunks)])                 # each chunk is rendered separately
    return C.reshape(H, W, 3)

def densify(gaussians, grad_mag, budget, W):
    mu, log_s, theta, color, op_raw = [t.detach() for t in gaussians]
    keep = op_raw.sigmoid() >= prune_opacity
    mu, log_s, theta, color, op_raw, grad_mag = [t[keep] for t in (mu, log_s, theta, color, op_raw, grad_mag)]
    dense = grad_mag > grad_threshold
    room = max(budget - mu.shape[0], 0)
    if dense.sum() > room:
        idx = torch.where(dense)[0]
        top = idx[grad_mag[idx].topk(room).indices]
        dense = torch.zeros_like(dense)
        dense[top] = True
    max_scale = log_s.exp().max(dim=1).values
    clone = dense & (max_scale <= size_threshold * W)
    split = dense & (max_scale >  size_threshold * W)
    children = []
    for i in range(2):
        z = torch.randn_like(mu[split]) * log_s[split].exp()
        c, s = torch.cos(theta[split]), torch.sin(theta[split])
        offset = torch.stack([c * z[:, 0] - s * z[:, 1],
                              s * z[:, 0] + c * z[:, 1]], -1)
        children.append([mu[split] + offset, log_s[split] - torch.log(torch.tensor(split_scale)),
                theta[split], color[split], op_raw[split]])
    child1, child2 = children[0], children[1]

    old    = [t[~split] for t in (mu, log_s, theta, color, op_raw)]
    clones = [t[clone]  for t in (mu, log_s, theta, color, op_raw)]

    return tuple(torch.cat([o, k, c1, c2]).requires_grad_()
                 for o, k, c1, c2 in zip(old, clones, child1, child2))





def quaternion_to_rotation(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),], -2)
    return R

def covariance_3d(scale, quat):
    my_matrix = quaternion_to_rotation(quat)
    my_matrix = my_matrix * scale[:, None, :]
    return (my_matrix @ my_matrix.transpose(-1, -2))

def project_gaussian(mu3, Sigma3, R_wc, t, K):
    mu_cam = mu3 @ R_wc.T + t
    x, y, z = mu_cam.unbind(-1)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    mu2 = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)

    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zero,   -fx * x / z**2], -1),
        torch.stack([zero,   fy / z, -fy * y / z**2], -1)], -2)

    Scam = R_wc @ Sigma3 @ R_wc.T
    Sig2 = J @ Scam @ J.transpose(-1, -2)
    return mu2, Sig2, z

def load_cameras(split):
    with open(f"spheres/cameras.json") as f:
        meta = json.load(f)
    K = torch.tensor(meta["K"], dtype=torch.float32, device=dev)
    cams = []
    for frame in meta["frames"] + meta.get("val_frames", []):
        if not frame["file"].startswith(split + "/"):
            continue
        img_np = np.asarray(Image.open(f"spheres/{frame['file']}").convert("RGB"), dtype=np.float32) / 255
        cams.append(SimpleNamespace(
            R=torch.tensor(frame["R_wc"], dtype=torch.float32, device=dev),
            t=torch.tensor(frame["t"],    dtype=torch.float32, device=dev),
            K=K,
            image=torch.from_numpy(img_np).to(dev),
            H=meta["height"], W=meta["width"]))
    return cams

def render_view(cam, mu3, log_s, quat, color, op_raw):
    Sig3 = covariance_3d(log_s.exp(), quat)
    mu2, Sig2, depth = project_gaussian(mu3, Sig3, cam.R, cam.t, cam.K)
    order = torch.argsort(depth, descending=False)
    return render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, cam.H, cam.W)

def mean_psnr(cams, mu3, log_s, quat, color, op_raw):
    total = 0.0
    with torch.no_grad():
        for cam in cams:
            img = render_view(cam, mu3, log_s, quat, color, op_raw)
            total += (-10 * torch.log10(((img - cam.image) ** 2).mean())).item()
    total /= len(cams)
    return total

def save_views(cams, idxs, tag, mu3, log_s, quat, color, op_raw):
    fig, ax = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)))
    with torch.no_grad():
        for row, i in enumerate(idxs):
            img = render_view(cams[i], mu3, log_s, quat, color, op_raw).clamp(0, 1).cpu().numpy()
            ax[row, 0].imshow(cams[i].image.cpu().numpy())
            ax[row, 0].set_title(f"view {i} ground truth")
            ax[row, 1].imshow(img)
            ax[row, 1].set_title(f"view {i} render")
            for a in ax[row]: a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{tag}.png", dpi=150)
    plt.close(fig)

def fit(path, budget, use_densify=False, steps=2000):
    torch.manual_seed(0)                                 # same init for every run
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    target = torch.from_numpy(img_np).to(dev)
    H, W = target.shape[:2]
    N = budget // 4 if use_densify else budget

    mu     = (torch.rand(N, 2, device=dev) * torch.tensor([W, H], device=dev)).requires_grad_()
    log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2, device=dev)).requires_grad_()
    theta  = torch.zeros(N, device=dev, requires_grad=True)
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)

    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    grad_accum = torch.zeros(N, device=dev)
    steps_since = 0

    for step in range(steps):
        depth_order = torch.arange(mu.shape[0], device=dev)
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order, H, W)
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        grad_accum += mu.grad.norm(dim=1)
        steps_since += 1
        opt.step()

        if use_densify and (step + 1) % densify_every == 0 and step < int(0.75 * steps):
            g = grad_accum / steps_since
            mu, log_s, theta, color, op_raw = densify((mu, log_s, theta, color, op_raw), g, budget, W)
            opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
            grad_accum = torch.zeros(mu.shape[0], device=dev)
            steps_since = 0

    # final PSNR, measured the same way for every run (after the last update)
    with torch.no_grad():
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(),
                       torch.arange(mu.shape[0], device=dev), H, W)
        psnr  = (-10 * torch.log10(((img - target) ** 2).mean())).item()

    out = img.clamp(0, 1).cpu().numpy()
    err = np.abs(out - img_np).mean(axis=2)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img_np);            ax[0].set_title("target")
    ax[1].imshow(out);               ax[1].set_title(f"render (N={mu.shape[0]}, {psnr:.2f} dB)")
    ax[2].imshow(err, cmap="magma"); ax[2].set_title("abs error")
    for a in ax: a.axis("off")
    plt.tight_layout()
    name = path.rsplit(".", 1)[0]
    plt.savefig(f"compare_{name}_{budget}{'_dense' if use_densify else ''}.png", dpi=150)
    plt.close(fig)

    return psnr

def densify_3d(gaussians, grad_mag, budget):
    mu, log_s, theta, color, op_raw = [t.detach() for t in gaussians]
    keep = op_raw.sigmoid() >= prune_opacity
    keep_idx = torch.where(keep)[0]
    mu, log_s, theta, color, op_raw, grad_mag = [t[keep] for t in (mu, log_s, theta, color, op_raw, grad_mag)]
    dense = grad_mag > grad_threshold
    room = max(budget - mu.shape[0], 0)
    if dense.sum() > room:
        idx = torch.where(dense)[0]
        top = idx[grad_mag[idx].topk(room).indices]
        dense = torch.zeros_like(dense)
        dense[top] = True
    max_scale = log_s.exp().max(dim=1).values
    clone = dense & (max_scale <= size_threshold)
    split = dense & (max_scale >  size_threshold)
    children = []
    for i in range(2):
        z = torch.randn_like(mu[split]) * log_s[split].exp()
        offset = (quaternion_to_rotation(theta[split]) @ z[:, :, None])
        offset = offset[:, :, 0]
        children.append([mu[split] + offset, log_s[split] - torch.log(torch.tensor(split_scale)),
                theta[split], color[split], op_raw[split]])
    child1, child2 = children[0], children[1]
    old    = [t[~split] for t in (mu, log_s, theta, color, op_raw)]
    clones = [t[clone]  for t in (mu, log_s, theta, color, op_raw)]
    return tuple(torch.cat([o, k, c1, c2]).requires_grad_() for o, k, c1, c2 in zip(old, clones, child1, child2)), keep_idx[~split]

def rebuild(old_opt, old_params, new_params, survivors, lr=1e-2):
    new_opt = torch.optim.Adam(new_params, lr=lr)
    for p_old, p_new in zip(old_params, new_params):
        state = old_opt.state.get(p_old)
        if not state:
            continue
        exp_avg    = torch.zeros_like(p_new)
        exp_avg_sq = torch.zeros_like(p_new)
        exp_avg[:len(survivors)]    = state["exp_avg"][survivors]
        exp_avg_sq[:len(survivors)] = state["exp_avg_sq"][survivors]
        new_opt.state[p_new] = {"step": state["step"].clone(), "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    return new_opt


os.makedirs("results", exist_ok=True)

RUN_P4 = True
RUN_P5 = True
RUN_P7 = True
RUN_P8 = True
RUN_P9 = True

images = ["coffee", "astronaut", "cat"]
iters = 1500
val_views = [0, 6]

if RUN_P4:
    grad_threshold = 2e-4
    p4_budget = 1024
    p4 = {}
    for name in images:
        p4[name] = {"plain":     fit(f"{name}.png", p4_budget, use_densify=False),
                    "densified": fit(f"{name}.png", p4_budget, use_densify=True)}
        print(f"P4 {name:10s} plain={p4[name]['plain']:.2f}  densified={p4[name]['densified']:.2f}")
        torch.cuda.empty_cache()
    with open("results/p4_results.json", "w") as f:
        json.dump({"budget": p4_budget, "psnr": p4}, f, indent=2)

if RUN_P5:
    counts = [256, 1024, 4096]
    results = {}
    for name in images:
        results[name] = []
        for N in counts:
            psnr = fit(f"{name}.png", N)
            results[name].append(psnr)
            print(f"P5 {name:10s} N={N:5d}  size={N * 9:6d}  PSNR={psnr:.2f} dB")
            torch.cuda.empty_cache()
    with open("results/p5_results.json", "w") as f:
        json.dump({"counts": counts, "psnr": results}, f, indent=2)

    plt.figure(figsize=(6, 4))
    for name in images:
        plt.plot(counts, results[name], marker="o", label=name)
    plt.xscale("log", base=2)
    plt.xticks(counts, [str(n) for n in counts])
    plt.xlabel("number of Gaussians N")
    plt.ylabel("PSNR (dB)")
    plt.title("Quality vs. count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/psnr_vs_n.png", dpi=150)
    plt.close()

if RUN_P7 or RUN_P8 or RUN_P9:
    train_cameras = load_cameras("train")
    val_cameras   = load_cameras("val")
    print("cameras:", len(train_cameras), "train,", len(val_cameras), "val")

if RUN_P7:
    torch.manual_seed(0); random.seed(0)
    N = 4000
    mu3    = ((torch.rand(N, 3, device=dev) * 2 - 1) * 1.5).requires_grad_()
    log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev)).requires_grad_()
    quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0; quat.requires_grad_()
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)
    opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)

    for step in range(iters):
        cam  = random.choice(train_cameras)
        img  = render_view(cam, mu3, log_s, quat, color, op_raw)
        loss = ((img - cam.image) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print("P7", step, loss.item(), (-10 * torch.log10(loss)).item())

    p7 = {"count": mu3.shape[0],
          "train_psnr": mean_psnr(train_cameras, mu3, log_s, quat, color, op_raw),
          "val_psnr":   mean_psnr(val_cameras,   mu3, log_s, quat, color, op_raw),
          "init": "4000 Gaussians, means uniform in [-1.5,1.5]^3, scale 0.08, identity quat, gray, opacity ~0.12"}
    print("P7", p7)
    save_views(train_cameras, [0, len(train_cameras) // 3, 2 * len(train_cameras) // 3],
               "results/p7_train_views", mu3, log_s, quat, color, op_raw)
    save_views(val_cameras, val_views, "results/p7_val_views", mu3, log_s, quat, color, op_raw)
    torch.save({"mu3": mu3.detach(), "log_s": log_s.detach(), "quat": quat.detach(),
                "color": color.detach(), "op_raw": op_raw.detach()}, "results/p7_model.pt")
    with open("results/p7_results.json", "w") as f:
        json.dump(p7, f, indent=2)
    torch.cuda.empty_cache()

# ---------------- P8: densified 3D fit ----------------
if RUN_P8:
    grad_threshold = 2e-4       # 3D world units; retune from the printed quantiles
    torch.manual_seed(0); random.seed(0)
    budget = 4000                                         # same as P7
    N = budget // 4                                       # start small, grow toward the budget
    mu3    = ((torch.rand(N, 3, device=dev) * 2 - 1) * 1.5).requires_grad_()
    log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev)).requires_grad_()
    quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0; quat.requires_grad_()
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)
    opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)
    grad_accum = torch.zeros(N, device=dev)
    steps_since = 0

    for step in range(iters):
        cam  = random.choice(train_cameras)
        img  = render_view(cam, mu3, log_s, quat, color, op_raw)
        loss = ((img - cam.image) ** 2).mean()
        opt.zero_grad(); loss.backward()
        grad_accum += mu3.grad.norm(dim=1)
        steps_since += 1
        opt.step()

        if (step + 1) % densify_every == 0 and step < int(0.75 * iters):
            g = grad_accum / steps_since
            print("P8 grad quantiles (50/90/99%):", torch.quantile(g, torch.tensor([0.5, 0.9, 0.99], device=dev)).tolist())
            old_params = [mu3, log_s, quat, color, op_raw]
            (mu3, log_s, quat, color, op_raw), survivors = densify_3d(tuple(old_params), g, budget)
            opt = rebuild(opt, old_params, [mu3, log_s, quat, color, op_raw], survivors)
            grad_accum = torch.zeros(mu3.shape[0], device=dev)
            steps_since = 0

        if step % 100 == 0:
            print("P8", step, mu3.shape[0], loss.item(), (-10 * torch.log10(loss)).item())

    p8 = {"count": mu3.shape[0],
          "train_psnr": mean_psnr(train_cameras, mu3, log_s, quat, color, op_raw),
          "val_psnr":   mean_psnr(val_cameras,   mu3, log_s, quat, color, op_raw)}
    save_views(val_cameras, val_views, "results/p8_val_views", mu3, log_s, quat, color, op_raw)
    torch.save({"mu3": mu3.detach(), "log_s": log_s.detach(), "quat": quat.detach(),
                "color": color.detach(), "op_raw": op_raw.detach()}, "results/p8_model.pt")
    with open("results/p8_results.json", "w") as f:
        json.dump(p8, f, indent=2)

    with open("results/p7_results.json") as f:
        p7 = json.load(f)
    print(f"plain     N={p7['count']}  train={p7['train_psnr']:.2f}  val={p7['val_psnr']:.2f}")
    print(f"densified N={p8['count']}  train={p8['train_psnr']:.2f}  val={p8['val_psnr']:.2f}")
    torch.cuda.empty_cache()

if RUN_P9:
    with open("results/p7_results.json") as f:
        p7 = json.load(f)
    m = torch.load("results/p8_model.pt")
    params = (m["mu3"], m["log_s"], m["quat"], m["color"], m["op_raw"])

    per_view = []
    with torch.no_grad():
        for cam in val_cameras:
            img = render_view(cam, *params)
            per_view.append((-10 * torch.log10(((img - cam.image) ** 2).mean())).item())
    save_views(val_cameras, list(range(len(val_cameras))), "results/p9_all_val_views", *params)
    base = val_cameras[0]
    frames = []
    with torch.no_grad():
        for k in range(36):
            phi = 2 * np.pi * k / 36
            Ry = torch.tensor([[np.cos(phi), 0, np.sin(phi)],
                               [0,           1, 0          ],
                               [-np.sin(phi), 0, np.cos(phi)]], dtype=torch.float32, device=dev)
            cam = SimpleNamespace(R=base.R @ Ry, t=base.t, K=base.K, H=base.H, W=base.W)
            img = render_view(cam, *params).clamp(0, 1).cpu().numpy()
            frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    frames[0].save("results/p9_orbit.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
    for k in [0, 9, 18, 27]:
        frames[k].save(f"results/p9_orbit_{k:02d}.png")

    p9 = {"val_psnr_per_view": per_view,
          "val_psnr_mean": sum(per_view) / len(per_view),
          "p7_train_psnr": p7["train_psnr"],
          "p7_val_psnr": p7["val_psnr"]}
    print("P9", p9)
    with open("results/p9_results.json", "w") as f:
        json.dump(p9, f, indent=2)
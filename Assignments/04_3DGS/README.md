# Assignment 4 - Implement Simplified 3D Gaussian Splatting

### In this assignment, you will implement a simplified version of 3D Gaussian Splatting in pure PyTorch. The pipeline reconstructs a scene from multi-view images by initializing 3D Gaussians from COLMAP points and optimizing them with differentiable rendering.

### Resources:
- [Paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf)
- [3DGS Official Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP Structure-from-Motion](https://colmap.github.io/)
- [Teaching Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Implementation of Simplified 3DGS

This repository is Yuxuan Song's implementation of Assignment_04 of DIP. The assignment contains three parts:

- Running COLMAP to recover camera parameters and a sparse 3D point cloud.
- Implementing a differentiable 3D Gaussian renderer in PyTorch.
- Training Gaussian parameters and comparing the simplified implementation with the official 3DGS system.

The repository provides two NeRF-synthetic style multi-view scenes, `chair` and `lego`.

<img src="pics/scene_preview.png" alt="scene preview" width="800">

## Data

The dataset is stored under `data/`:

```text
data/
|-- chair/images/   # 100 multi-view rendered images, 800 x 800
|-- lego/images/    # 100 multi-view rendered images, 400 x 400
```

The local data statistics are:

- `chair`: 100 RGBA images, each `800 x 800`
- `lego`: 100 RGBA images, each `400 x 400`

Before training, COLMAP must be run for the selected scene. The training code expects the following sparse reconstruction files:

```text
<scene>/
|-- images/
|-- sparse/
    |-- 0_text/
        |-- cameras.txt
        |-- images.txt
        |-- points3D.txt
```

The `ColmapDataset` in `data_utils.py` reads these text files and converts COLMAP cameras, poses, 3D points, and RGB colors into tensors for training.

## Requirements

To install Python dependencies:

```setup
python -m pip install torch torchvision opencv-python numpy pillow tqdm natsort
```

The implementation also uses PyTorch3D for KNN-based scale initialization:

```setup
python -m pip install pytorch3d
```

COLMAP is required for camera pose estimation and sparse reconstruction. On Linux, it can be installed with:

```setup
conda install -c conda-forge colmap
```

CUDA is strongly recommended for training because the renderer evaluates all Gaussians on the full image grid.

## Running

To run COLMAP on the `chair` scene:

```colmap
python mvs_with_colmap.py --data_dir data/chair
```

This script performs feature extraction, exhaustive matching, sparse reconstruction, and text-format model conversion. The final text model is saved to:

```text
data/chair/sparse/0_text/
```

To check the COLMAP result by projecting sparse 3D points back to the training views:

```debug
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

The debug script saves side-by-side visualizations to:

```text
data/chair/projections/
```

To train the simplified 3DGS model:

```train
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

Useful optional arguments:

```train
python train.py \
    --colmap_dir data/chair \
    --checkpoint_dir data/chair/checkpoints \
    --num_epochs 200 \
    --batch_size 1 \
    --debug_every 1 \
    --debug_samples 4 \
    --device cuda
```

To render an orbit video from a trained checkpoint:

```render
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000060.pt \
    --output data/chair/render_mv.mp4 \
    --num_frames 240 \
    --fps 30
```

## Method

### Task 1: Structure-from-Motion with COLMAP

The script `mvs_with_colmap.py` runs a sparse COLMAP pipeline:

1. `feature_extractor` extracts SIFT features from all input images.
2. `exhaustive_matcher` matches features between image pairs.
3. `mapper` estimates camera poses and triangulates sparse 3D points.
4. `model_converter` converts COLMAP binary outputs to text files.

The script uses a shared `PINHOLE` camera model because all views in a scene are rendered with the same camera intrinsics. GPU SIFT is disabled in the script with `--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0`, which improves portability on headless servers.

The script `debug_mvs_by_projecting_pts.py` validates the COLMAP reconstruction. It reads `cameras.txt`, `images.txt`, and `points3D.txt`, projects the sparse 3D points into each image using the recovered intrinsics and extrinsics, then saves a side-by-side image containing the original view and the projected point visualization.

### Task 2: Gaussian Model

The Gaussian representation is implemented in `gaussian_model.py`. Each COLMAP 3D point is converted into one learnable 3D Gaussian. The optimized parameters are:

```text
position:  (N, 3)
rotation:  (N, 4), quaternion
scale:     (N, 3), stored in log space
color:     (N, 3), stored in logit space
opacity:   (N, 1), stored in logit space
```

Positions are initialized directly from `points3D.txt`. Colors are initialized from COLMAP point colors and converted to logits so that `sigmoid(colors)` remains in `[0, 1]`. Opacities are initialized to `8.0` in logit space, so the initial opacity is close to one.

Scales are initialized from local point density. The code uses PyTorch3D `knn_points` to compute distances to nearby points. The mean distance to the nearest neighbors is multiplied by 2 and clamped around the median value to avoid extremely small or large initial Gaussians.

The 3D covariance matrix is built from quaternion rotation and scale:

```text
Sigma = R S S^T R^T
```

The quaternion is normalized before conversion to a rotation matrix. The scale vector is exponentiated from log space and placed on the diagonal of `S`.

### Task 2: Differentiable Renderer

The renderer is implemented in `gaussian_renderer.py`.

#### 3D to 2D Projection

For each view, world-space Gaussian centers are transformed to camera space:

```text
X_cam = X_world R^T + t
```

The projected 2D mean is computed with the camera intrinsic matrix:

```text
screen = X_cam K^T
mu_2D = screen_xy / z
```

To project each 3D covariance into image space, the renderer computes the Jacobian of perspective projection and applies:

```text
Sigma_2D = J R Sigma_3D R^T J^T
```

This gives every 3D Gaussian an elliptical 2D footprint on the image plane.

#### 2D Gaussian Evaluation

For every pixel, the renderer evaluates the projected Gaussian density:

```text
G(x) = 1 / (2 pi sqrt(det(Sigma_2D))) *
       exp(-0.5 * (x - mu)^T Sigma_2D^{-1} (x - mu))
```

The implementation symmetrizes the covariance matrix and adds a small diagonal epsilon before inversion for numerical stability. The exponent is also clamped to avoid overflow.

#### Alpha Compositing

Gaussians are sorted by depth. For each pixel, alpha is computed as:

```text
alpha_i = opacity_i * G_i(x)
```

The transmittance is accumulated front-to-back:

```text
T_i = product_{j < i} (1 - alpha_j)
weight_i = alpha_i * T_i
```

The final pixel color is:

```text
C = sum_i weight_i * color_i
```

The output is clamped to `[0, 1]`.

### Training

The training loop is implemented in `train.py`. It loads one COLMAP view per batch, renders the current Gaussian model from that view, and minimizes an L1 RGB reconstruction loss:

```text
loss = mean(abs(rendered_image - ground_truth_image))
```

Images are downsampled by a factor of 8 in `ColmapDataset`, so the `chair` training resolution becomes `100 x 100`. This keeps the pure PyTorch renderer tractable.

The optimizer uses separate learning rates for different Gaussian parameters:

```text
position:  0.000016
color:     0.025
opacity:   0.05
scale:     0.005
rotation:  0.001
```

Checkpoints are saved every 20 epochs. Debug images are saved every epoch by rendering several fixed validation views. Each debug image contains:

```text
top row:    ground truth views
bottom row: rendered views
```

After training, the script also creates `debug_rendering.mp4`, which compares each training view with the corresponding rendered image.

### Multi-view Rendering

The optional script `render_3dgs_mv.py` renders a horizontal orbit video from a trained checkpoint. It estimates:

- scene center from the centroid of COLMAP 3D points
- up direction from the average camera down direction
- orbit radius and elevation from training camera centers

It then builds a COLMAP/OpenCV-style look-at camera path and renders each frame with the trained Gaussian model.

## Results

### COLMAP Reconstruction

COLMAP provides the sparse point cloud and camera parameters required to initialize 3DGS. The sparse points are not dense enough for direct high-quality rendering, but they provide good initial Gaussian centers and colors.

The projection debug step is useful because it verifies that `points3D.txt`, `images.txt`, and `cameras.txt` are interpreted consistently. If projected sparse points align with the input image, the camera convention and intrinsic matrix are correct.

### Simplified 3DGS Rendering

The PyTorch renderer can optimize a Gaussian cloud to reproduce the training views. The result quality mainly depends on:

- quality and density of COLMAP sparse points
- stability of covariance projection
- correct depth ordering
- scale initialization
- number of optimization epochs

Because this is a simplified implementation, it is expected to produce blurrier and less complete results than the official 3DGS code. The implementation does not include adaptive densification, pruning, spherical harmonics, tile-based rasterization, or the optimized CUDA rasterizer.

### Comparison with Official 3DGS

Compared with the official 3DGS implementation:

| Aspect | This implementation | Official 3DGS |
|---|---|---|
| Renderer | Pure PyTorch, full image grid | CUDA tile-based rasterizer |
| Appearance | RGB color per Gaussian | Spherical harmonics view-dependent color |
| Point growth | Fixed COLMAP points | Adaptive densification and pruning |
| Speed | Slow, memory-heavy | Real-time or near real-time rendering |
| Quality | Coarser and blurrier | Sharper, denser, more complete |
| Educational value | Transparent and easy to inspect | Production-quality but more complex |

The official version is substantially faster because it avoids evaluating every Gaussian at every pixel. Instead, it bins Gaussians into screen-space tiles and only splats local contributions. It also improves geometry during training by splitting or cloning Gaussians in high-gradient regions and pruning unimportant ones.

## Analysis

### Numerical Stability

The renderer uses covariance symmetrization, diagonal epsilon, determinant clamping, exponent clamping, and alpha clamping to improve stability. These details are important because covariance projection can produce nearly singular 2D matrices.

One subtle issue is depth handling. The current renderer builds a `valid_mask` for positive-depth points, but the code evaluates all projected Gaussians before applying the mask. A safer implementation would filter invalid-depth Gaussians before computing Gaussian values and matrix inverses. This avoids `NaN` propagation from points behind the camera.

### Why the Simplified Renderer Is Slow

The renderer constructs tensors with shape roughly `(N, H, W)`. Even after downsampling to `100 x 100`, a scene with thousands of Gaussians still creates large intermediate tensors. This is easy to understand and differentiable in PyTorch, but it is not scalable.

Official 3DGS avoids this by using a custom rasterizer that only evaluates Gaussians near their projected support. This is the main reason for the speed gap.

### Why COLMAP Initialization Matters

3DGS training is highly dependent on initialization. If COLMAP produces wrong camera poses or sparse points, Gaussians will project to the wrong locations and optimization may fail. The provided projection debug tool is therefore an important sanity check before training.

Good scale initialization is also important. If the Gaussians are too small, the rendered image is sparse and gradients are weak. If they are too large, the rendered result becomes overly smooth and depth ordering artifacts become stronger.

## Limitations and Future Work

The current implementation is intentionally simple. Future improvements include:

- filtering invalid-depth points before covariance inversion
- removing the Gaussian PDF normalizer for a more splat-like alpha footprint
- adding image-space culling to skip off-screen Gaussians
- adding adaptive densification and pruning
- using spherical harmonics for view-dependent color
- replacing full-grid PyTorch rasterization with tile-based rasterization
- adding SSIM or perceptual loss in addition to L1 RGB loss

## Acknowledgement

> Thanks for the algorithms proposed by [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf), the [official 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting), and [COLMAP](https://colmap.github.io/).

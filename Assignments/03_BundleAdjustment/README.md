# Assignment 3 - Bundle Adjustment

### In this assignment, you will implement Bundle Adjustment from scratch with PyTorch, and use COLMAP to perform a complete multi-view 3D reconstruction pipeline.

### Resources:
- [Teaching Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)
- [Bundle Adjustment - Wikipedia](https://en.wikipedia.org/wiki/Bundle_adjustment)
- [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)
- [pytorch3d.transforms](https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html)
- [COLMAP Documentation](https://colmap.github.io/)
- [COLMAP Tutorial](https://colmap.github.io/tutorial.html)

## Implementation of Bundle Adjustment

This repository is Yuxuan Song's implementation of Assignment_03 of DIP. The assignment contains two parts:

- A PyTorch implementation of Bundle Adjustment from 2D point observations.
- A COLMAP command-line reconstruction pipeline for sparse and dense reconstruction.

The provided dataset contains a synthetic multi-view capture of a human head and upper body. The goal of Task 1 is to recover the 3D point cloud, camera extrinsics, and shared focal length only from 2D observations. Task 2 reconstructs the same scene from rendered images using COLMAP.

<img src="pics/data_overview.png" alt="data overview" width="800">

## Data

The dataset is stored in `data/`:

```text
data/
|-- images/              # 50 rendered views, 1024 x 1024
|-- points2d.npz         # 2D observations for all views
|-- points3d_colors.npy  # RGB color for each 3D point
```

The `points2d.npz` file contains 50 keys from `view_000` to `view_049`. Each view stores an array of shape `(20000, 3)`, where every row is:

```text
x, y, visibility
```

- `x, y`: observed 2D pixel coordinates.
- `visibility`: `1.0` for visible points and `0.0` for occluded points.

The dataset statistics are:

- Number of views: `50`
- Number of points: `20000`
- Image size: `1024 x 1024`
- Total visible observations: `805089`
- Average visible points per view: about `16102`
- Minimum / maximum visible points per view: `11114 / 19550`

The color file `points3d_colors.npy` has shape `(20000, 3)` and stores RGB values in `[0, 1]` for visualizing the reconstructed point cloud.

## Requirements

To install the Python dependencies for Task 1:

```setup
python -m pip install torch numpy matplotlib opencv-python
```

For Task 2, COLMAP is required. On Linux, it can be installed with conda:

```setup
conda install -c conda-forge colmap
```

Dense COLMAP reconstruction is much faster with CUDA support.

## Running

To visualize the provided images and 2D observations, run:

```visualize
python visualize_data.py
```

The script overlays visible 2D points on five selected views and saves the results to `data/vis/`.

To run the PyTorch Bundle Adjustment implementation, run:

```ba
python task1.py --data_dir data --out_dir outputs/ba_task1 --steps 5000 --batch_points 4096
```

The script saves:

```text
outputs/ba_task1/
|-- reconstructed_points.obj
|-- reconstructed_points.ply
|-- ba_result.npz
|-- loss_curve.png
|-- loss_log.csv
```

To run the COLMAP reconstruction pipeline, run:

```colmap
bash run_colmap.sh
```

The script saves sparse reconstruction results to `data/colmap/sparse/0/` and dense fused point cloud results to `data/colmap/dense/fused.ply`.

## Method

### Task 1: PyTorch Bundle Adjustment

The PyTorch BA implementation optimizes the following unknowns jointly:

- 3D point coordinates: `points3d`, shape `(N, 3)`
- Camera rotations: Euler angles, shape `(V, 3)`
- Camera translations: `translations`, shape `(V, 3)`
- Shared focal length: one scalar parameter

The focal length is initialized from a 60 degree field of view:

```text
f = H / (2 * tan(fov / 2))
```

The implementation stores the focal length through a raw learnable parameter and converts it with `softplus`, which keeps the focal length positive during optimization.

#### Camera Model

The coordinate system follows the assignment convention. The object is near the origin and the camera observes it from the positive Z side. Camera coordinates are computed as:

```text
[Xc, Yc, Zc] = R @ [X, Y, Z] + T
```

The camera translation is initialized as `[0, 0, -2.5]`, so valid points usually have negative camera-space depth. The projection formula is:

```text
u = -f * Xc / Zc + cx
v =  f * Yc / Zc + cy
```

where `cx = image_width / 2` and `cy = image_height / 2`.

<img src="pics/coordinate_system.png" alt="coordinate system" width="500">

#### Rotation Parameterization

Camera rotations are represented by XYZ Euler angles. The code implements axis-angle rotation matrices for X, Y, and Z axes, then composes them as:

```text
R = Rx @ Ry @ Rz
```

This avoids optimizing a full unconstrained `3 x 3` matrix and keeps the rotation representation compact.

#### Initialization

The 3D points are initialized from the average 2D observation of each point. For each point, the code averages visible image coordinates across views, back-projects them approximately with the initialized focal length and camera distance, then adds a small random Z perturbation.

Camera yaw angles are initialized linearly from `-70` degrees to `+70` degrees, matching the known multi-view capture range. All translations are initialized at the same camera distance along the negative Z direction.

#### Loss Function

The main objective is the reprojection loss between projected 3D points and observed 2D points:

```text
loss_reproj = mean(||projected_xy - observed_xy||^2)
```

Only visible observations are used in the loss. The implementation normalizes the squared pixel error by image size and also reports RMSE in pixels for easier interpretation.

The optimization includes several stabilizing regularizers:

- Pose prior: keeps rotations, translations, and focal length close to their initial values.
- Point center prior: keeps the reconstructed point cloud centered.
- Depth penalty: discourages points from crossing behind the expected valid depth range.

The model is optimized with Adam using separate learning rates for points, poses, and focal length:

```text
lr_points = 1e-2
lr_pose   = 3e-3
lr_focal  = 1e-3
```

For memory efficiency, the code samples a subset of points at each step. The default mini-batch contains `4096` points, but the full RMSE is evaluated periodically.

#### Output

After optimization, the code exports the reconstructed point cloud in both OBJ and PLY formats. Each vertex stores position and RGB color:

```text
v x y z r g b
```

The script also stores the learned camera parameters and focal length in `ba_result.npz`, writes the loss history to `loss_log.csv`, and plots `loss_curve.png`.

### Task 2: COLMAP Reconstruction

The COLMAP pipeline is implemented in `run_colmap.sh`. It performs the standard reconstruction steps:

1. Feature extraction with the PINHOLE camera model.
2. Exhaustive feature matching.
3. Sparse reconstruction with COLMAP mapper.
4. Image undistortion for dense reconstruction.
5. PatchMatch stereo.
6. Stereo fusion into a dense point cloud.

The script creates a COLMAP workspace under `data/colmap/`:

```text
data/colmap/
|-- database.db
|-- sparse/
|-- dense/
```

The sparse reconstruction estimates camera poses and sparse 3D points from matched image features. The dense pipeline then computes stereo depth maps and fuses them into `fused.ply`.

## Results

### Bundle Adjustment Result

The expected BA output is a colored point cloud of the original model. The provided result visualization shows that the optimized points recover the human head and upper-body geometry from only 2D observations.

<img src="pics/result.gif" alt="bundle adjustment result" width="300">

The important observations are:

- The reprojection loss directly measures consistency between the recovered 3D structure and all visible 2D observations.
- Jointly optimizing points, poses, and focal length makes the problem flexible, but also introduces gauge ambiguity.
- Initialization is important. A reasonable camera distance, focal length, and yaw range make the optimization converge much more reliably.
- Visibility masking is necessary because occluded points should not contribute to reprojection loss.

### COLMAP Result

COLMAP reconstructs the scene from rendered images rather than from the provided 2D correspondence file. Its sparse reconstruction internally performs feature matching and Bundle Adjustment, while the dense stage recovers a more complete point cloud.

The COLMAP pipeline is useful as a practical comparison to Task 1:

- Task 1 uses known point identities across all views and optimizes directly from 2D tracks.
- COLMAP starts only from images, extracts and matches features, estimates cameras, triangulates points, and refines everything with BA.

Thus, Task 1 focuses on the mathematical core of Bundle Adjustment, while Task 2 demonstrates how the same idea is used inside a full reconstruction system.

## Analysis

### Reprojection Optimization

Bundle Adjustment is a nonlinear least-squares problem. The projection operation couples every 3D point with every camera that observes it, so errors in camera pose and errors in point position can compensate for each other. The implementation reduces this difficulty by using good initialization and weak priors.

Using mini-batches of points makes training practical for 20000 points and 50 views. Because all cameras are still used for each sampled point batch, the optimization still receives multi-view geometric constraints at every step.

The sign convention of depth is the most important implementation detail. In this coordinate system, valid camera-space points have negative `Zc`, so the projection uses `-f * Xc / Zc` for horizontal coordinates. This avoids left-right flipping.

### COLMAP Pipeline

The COLMAP script follows a complete Structure-from-Motion and Multi-View Stereo pipeline. Feature extraction and matching produce 2D correspondences automatically. The mapper estimates sparse geometry and camera poses, then dense reconstruction uses PatchMatch stereo and stereo fusion.

Compared with the PyTorch BA task, COLMAP is more automatic and robust, but it is less transparent as a learning exercise. The PyTorch implementation makes every optimized variable and loss term explicit, which is useful for understanding how 3D structure is recovered from 2D observations.

## Limitations and Future Work

The current PyTorch BA implementation uses Euler angles, which are simple but can suffer from singularities. A future version could use Lie algebra `se(3)` updates or quaternions for more stable pose optimization.

The loss currently uses plain squared reprojection error. Robust losses such as Huber loss or Geman-McClure loss could improve stability when observations contain outliers.

The COLMAP script uses exhaustive matching, which is reasonable for 50 images but does not scale well to large datasets. For larger scenes, sequential matching or vocabulary-tree matching would be more efficient.

## Acknowledgement

> Thanks for the classical Bundle Adjustment formulation and for the reconstruction pipeline provided by [COLMAP](https://colmap.github.io/).

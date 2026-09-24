# Comprehensive Research & Literature Survey
## Mathematical Foundations, AI SOTA, and Photogrammetric Architecture for SIH26158

---

## 1. Physics & Geometry of Single-Pass Drone Videography

### 1.1 The Single-Pass Parallax Deficit

Traditional aerial photogrammetry relies on multi-pass "lawnmower" flight paths with $70\text{--}85\%$ forward overlap and $60\text{--}70\%$ lateral (side) overlap. This guarantees that every ground point $X \in \mathbb{R}^3$ is observed from multiple orthogonal baselines and diverse azimuth angles $\theta_i$.

In **single-pass UAV video** (SIH26158), the camera travels along a single continuous trajectory $T(t) \in \text{SE}(3)$. The fundamental physical bottlenecks include:

1. **Collinear Camera Centers**: The camera optical centers $C_1, C_2, \dots, C_N$ lie approximately along a 1D trajectory line. Baselines perpendicular to the flight path are near zero ($B_\perp \approx 0$).
2. **Depth Uncertainty Propagation**:
   The triangulation depth error $\sigma_Z$ from two views with baseline $B$, focal length $f$, and disparity error $\sigma_d$ is:
   $$\sigma_Z = \frac{Z^2}{B \cdot f} \sigma_d$$
   As baseline $B \to 0$, depth variance $\sigma_Z \to \infty$. In a single forward pass, consecutive frames have small baselines ($B \ll Z$), amplifying noise along the optical axis ($Z$).
3. **Doming / Bowling Deformation**: In linear flight strips without ground control points or cross-lines, small systematic errors in focal length estimation ($f$) or radial lens distortion ($k_1, k_2$) accumulate quadratically along the flight strip, curling planar ground into a cylindrical or parabolic bowl.
4. **Facade Blind Spots**: Nadir or fixed-pitch cameras view vertical building facades at glancing angles ($> 60^\circ$ from surface normal). Multi-view stereo (MVS) cannot correlate pixels across views where surface foreshortening approaches singularity.

---

### 1.2 Classical Structure-from-Motion (SfM) Mechanics

#### Two-View Epipolar Geometry & Essential Matrix
For calibrated normalized image coordinates $x_1, x_2 \in \mathbb{R}^3$:
$$x_2^T E x_1 = 0 \quad \text{where} \quad E = [t]_\times R$$
The 5-point algorithm with RANSAC recovers relative rotation $R \in \text{SO}(3)$ and unit translation direction $\hat{t} = t / \|t\| \in \mathbb{S}^2$. Notice that $\|t\|$ is scale-ambiguous; monocular vision cannot recover absolute physical scale without external priors.

#### Global Non-Linear Bundle Adjustment
COLMAP solves the joint non-linear least-squares optimization:
$$\min_{\{R_i, t_i\}, \{X_j\}} \sum_{i=1}^N \sum_{j \in \mathcal{V}_i} \rho\left( \left\| \pi(K_i, R_i X_j + t_i) - x_{ij} \right\|^2 \right)$$
where $\pi(\cdot)$ denotes perspective projection with radial distortion, $x_{ij}$ is the 2D feature observation of 3D point $X_j$ in camera $i$, and $\rho(\cdot)$ is the Cauchy robust loss function suppressing outlier correspondences.

---

## 2. Geodesy & Coordinate Transformations Without GCPs

SIH26158 mandates georeferenced output in a standard Coordinate Reference System (CRS) without manual Ground Control Points (GCPs). This requires rigorous geodesy:

```text
  [WGS-84 Ellipsoidal]         [Earth-Centered, Earth-Fixed]       [Local East-North-Up]
 (lat φ, lon λ, alt h)  ───►       (X_ecef, Y_ecef, Z_ecef)    ───►    (X_enu, Y_enu, Z_enu)
                                                                             │
                                                                             ▼
                                                                     [Local UTM Zone]
                                                                     (Easting, Northing, Elev)
```

### 2.1 Closed-Form Geodetic Transformations

#### WGS-84 to ECEF:
Given equatorial radius $a = 6378137.0\text{ m}$ and first eccentricity squared $e^2 = 0.00669437999014$:
$$\nu(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$
$$X = (\nu + h) \cos\phi \cos\lambda$$
$$Y = (\nu + h) \cos\phi \sin\lambda$$
$$Z = \left( (1 - e^2)\nu + h \right) \sin\phi$$

#### ECEF to Local ENU:
Centered at scene anchor $(\phi_0, \lambda_0, h_0)$:
$$\begin{bmatrix} X_{\text{enu}} \\ Y_{\text{enu}} \\ Z_{\text{enu}} \end{bmatrix} =
\begin{bmatrix} 
-\sin\lambda_0 & \cos\lambda_0 & 0 \\
-\sin\phi_0\cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0 \\
\cos\phi_0\cos\lambda_0 & \cos\phi_0\sin\lambda_0 & \sin\phi_0
\end{bmatrix}
\begin{bmatrix} X - X_0 \\ Y - Y_0 \\ Z - Z_0 \end{bmatrix}$$

### 2.2 Metric Alignment: Umeyama Sim(3) vs GNSS-Aided Bundle Adjustment

1. **Loose Alignment (Umeyama Similarity Transform)**:
   Finds optimal scale $s \in \mathbb{R}^+$, rotation $R \in \text{SO}(3)$, and translation $t \in \mathbb{R}^3$ aligning SfM camera centers $\{C_i\}$ to geodetic positions $\{P_i\}$:
   $$\min_{s, R, t} \frac{1}{N} \sum_{i=1}^N \| P_i - (s R C_i + t) \|^2$$
   Solved in closed form via Singular Value Decomposition (SVD) of the cross-covariance matrix $\Sigma_{PC} = U \Sigma V^T$:
   $$R = U \begin{bmatrix} 1 & & \\ & 1 & \\ & & \det(U V^T) \end{bmatrix} V^T, \quad s = \frac{\text{Tr}(D \Sigma)}{\sigma_C^2}, \quad t = \mu_P - s R \mu_C$$
   *Limitation*: When the drone flies in a straight line, $\text{rank}(\Sigma_{PC}) \approx 1$, rendering cross-axis rotation unconstrained. DroneMap solves this by using flight trajectory curvature or falling back to GNSS heading priors.

2. **Tight Alignment (COLMAP `pose_prior_mapper`)**:
   Directly incorporates GPS priors into bundle adjustment:
   $$\min \sum \rho(\text{reprojection}) + \sum_{i} \frac{1}{2} (C_i - C_{i,\text{gps}})^T \Sigma_{\text{gps}}^{-1} (C_i - C_{i,\text{gps}})$$
   Jointly estimates internal camera calibration, metric scale, and scene geometry.

---

## 3. Deep Learning Feature Matching Survey

| Method | Feature Extractor | Matcher | Repeatability on Low Texture | Latency (Pair) | VRAM Footprint | License | Suitability for DroneMap |
|---|---|---|---|---|---|---|---|
| **SIFT (OpenCV/COLMAP)** | Difference-of-Gaussians | FLANN / Exhaustive L2 | Moderate (fails on uniform roofs/roads) | **~15 ms (CUDA)** | **< 500 MB** | BSD-3-Clause | **Default Workhorse**: Exact sub-pixel coordinates, zero hallucinations. |
| **SuperPoint + LightGlue** | FCN with synthetic pre-training | Deep Graph Transformer with self/cross attention | **Exceptional** (bridges large view angle shifts) | **~28 ms (CUDA)** | **~1.2 GB** | LightGlue: Apache-2.0; SP weights: Research | **Recommended Neural Matcher**: Injected into `colmap.db` via `hloc` conventions. |
| **LoFTR** | CNN + Transformer | Detector-free dense matching | High (dense coverage on textureless surfaces) | ~85 ms (CUDA) | ~3.5 GB | Apache-2.0 | Heavy VRAM; susceptible to edge floaters. |
| **RoMa** | DINOv2 + ViT | Dense robust matching with Markov Random Field | State-of-the-art on extreme viewpoint changes | ~180 ms (CUDA) | ~4.2 GB | MIT | Powerful but too slow for streaming 4K video. |
| **MASt3R** | Asymmetric ViT Backbone | Dual-branch dense point prediction | Jointly predicts 3D points and correspondences | ~350 ms (CUDA) | ~5.2 GB | CC BY-NC-SA 4.0 | High memory requirement; optimal for Cloud/Kaggle tier. |

---

## 4. Monocular Metric Depth Models

Monocular depth models estimate dense per-pixel distances from a single image.

### 4.1 Evaluation of Contemporary Backbones

1. **Depth Anything V2 (Yang et al., 2024)**:
   - *Architecture*: DINOv2 encoder + Dense Prediction Transformer (DPT) decoder trained on 62M synthetic and pseudo-labeled real images.
   - *Output*: Relative disparity $d \in [0, 1]$ with sharp object boundaries.
   - *Metric Conversion*: Scaled via telemetry altitude $h$ and camera pitch angle $\theta$:
     $$Z_{\text{metric}}(u, v) \approx \frac{h}{\cos\theta} \cdot \frac{d(u, v)}{\text{median}(d)}$$
   - *Role*: Ultra-crisp edge delineation for building outlines and occlusion boundaries.

2. **Metric3D v2 (Yin et al., 2024)**:
   - *Architecture*: ViT backbone with canonical camera transformation and learned focal length adaptation.
   - *Output*: Direct zero-shot absolute metric depth (in meters) without requiring external scaling.
   - *Role*: Provides an independent physical sanity check against consumer GPS vertical drift.

3. **Apple Depth Pro (Bochkovskii et al., 2024)**:
   - *Architecture*: Multi-scale ViT producing 2.25-megapixel metric depth maps in $0.3\text{ s}$.
   - *Output*: Metric depth + zero-shot focal length estimation.
   - *Role*: High-resolution depth maps without prior camera calibration.

---

## 5. Dynamic Object Segmentation & Transient Mitigation

SIH26158 mandates eliminating dynamic moving objects (vehicles, people, animals) to prevent reconstruction floaters and road artifacts.

### 5.1 Dynamic Masking Approaches

1. **YOLOv8s-seg (Current Production Standard)**:
   - Evaluates keyframes on CUDA in $\approx 12\text{ ms}$.
   - Targets 14 dynamic classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `train`, `truck`, `boat`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`.
   - Generates non-destructive binary PNG masks passed via COLMAP's `--ImageReader.mask_path`. SIFT keypoints within masked regions are suppressed before bundle adjustment.
2. **Velocity-Adaptive Dilation**:
   - Fixed dilation (12 px) fails on fast-moving cars whose motion blur extends 30–50 px.
   - Adaptive dilation computes the optical flow vector $\mathbf{v} = (v_x, v_y)$ at the object bounding box perimeter:
     $$r_{\text{dilate}} = r_{\text{base}} + \alpha \cdot \|\mathbf{v}\|$$
     where $\alpha = 0.5$, ensuring complete absorption of motion blur fringes.
3. **SAM 2 (Segment Anything 2)**:
   - Transformer memory architecture that tracks instances across video frames.
   - Ideal for complex scenes with intermittent occlusions (e.g. car driving under a tree canopy).

---

## 6. 3D Representations: Meshes vs Splats vs NeRF

| Metric / Dimension | Classical Multi-View Stereo (OpenMVS) | 2.5D Terrain Engine (`terrain.py`) | 3D Gaussian Splatting (3DGS) | Neural Radiance Fields (NeRF) |
|---|---|---|---|---|
| **Mathematical Representation** | Explicit Triangulated Mesh (TIN) | Watertight 2.5D Elevation Grid TIN | Unstructured 3D Gaussians $(\mu, \Sigma, c, \alpha)$ | Implicit MLP / Voxel Density Field $\sigma(x), c(x, d)$ |
| **Metric Accuracy** | High (derived from triangulated optical rays) | High (KDTree IDW on registered points) | Moderate (optimizes photometric loss, not metric coords) | Moderate (density volume optimized for photometric loss) |
| **GIS Compatibility (LAZ, DSM, DTM, Ortho)** | Native & Direct | Native & Direct | Requires rasterization and surface approximation | Requires marching cubes ray-marching |
| **CAD / Engineering Tools** | 100% standard OBJ / GLB | 100% standard OBJ / GLB | Incompatible with standard CAD/GIS | Incompatible with standard CAD/GIS |
| **Air-Gapped Web Viewport** | Standard Three.js WebGL (< 5 MB JS) | Standard Three.js WebGL (< 5 MB JS) | Heavy WebGL Gaussian shaders (high client VRAM) | Heavy neural client inference |
| **Local Runtime (RTX 3050)** | ~15–30 min | **~10–20 seconds** | ~20–35 min training | ~45–90 min training |

### Conclusion:
For defense intelligence and municipal GIS applications (SIH26158), **watertight polygon meshes and metric point clouds are indispensable**. 3D Gaussian Splatting is integrated as a complementary photorealistic novel-view visualizer in the cloud tier, but does not replace the metric GIS deliverables.

---

## 7. Key Literature Citations

1. Schönberger, J. L., & Frahm, J. M. (2016). *Structure-from-Motion Revisited*. CVPR 2016.
2. Cernea, D. (2020). *OpenMVS: Multi-View Stereo Reconstruction Library*.
3. Lindenberger, P., Sarlin, P. E., & Pollefeys, M. (2023). *LightGlue: Local Feature Matching at Light Speed*. ICCV 2023.
4. Sarlin, P. E., et al. (2019). *From Coarse to Fine: Robust Hierarchical Localization at Large Scale*. CVPR 2019.
5. Yang, L., et al. (2024). *Depth Anything V2: A More Capable Foundation Model for Monocular Depth Estimation*. NeurIPS 2024.
6. Yin, Y., et al. (2024). *Metric3D v2: A Versatile Camera-Calibrated Foundation Model for Monocular Metric Depth Estimation*. TPAMI 2024.
7. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023.
8. Guédon, A., & Lepetit, V. (2024). *SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and Editing*. CVPR 2024.
9. Umeyama, S. (1991). *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*. IEEE TPAMI 1991.

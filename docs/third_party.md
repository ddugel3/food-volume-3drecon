# Third-Party Requirements

Only project-owned Python files and newly prepared documentation are exported.
No vendored model repositories, weights, or competition data are included.
Review source ownership before assigning a license to the overall repository.

Components referenced by the implementation:

- [SAM](https://github.com/facebookresearch/segment-anything): check the terms
  for the specific SAM 3 / SAM 2 model and Transformers integration in use.
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO): optional
  alternative segmentation path.
- [Depth Pro](https://github.com/apple/ml-depth-pro): depth backbone and its
  relevant model/weight license.
- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1): separately
  acquired mesh-generation source and weights.
- [MoGe](https://github.com/microsoft/MoGe): optional depth backend.
- Qwen model ID `Qwen/Qwen3.8-27B-FP8` is retained from the original source;
  confirm model availability, access, and terms before using it.
- PyTorch, Transformers, NumPy, SciPy, Pillow, trimesh, scikit-learn, and optional
  OpenCV have their own package licenses and runtime requirements.

Download competition assets directly from
[Kaggle](https://www.kaggle.com/competitions/3d-reconstruction-from-monocular-multi-food-images/data)
after accepting the applicable terms. This repository does not redistribute
raw images, masks/visualizations derived from them, or GT volumes.

The architecture image is AI-generated explanatory artwork. The image's food,
depth, and mesh inserts are not experiment outputs or benchmark samples.

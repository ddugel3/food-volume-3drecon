# Reproduction Notes

## What Is Verified

The publication export contains the existing project-owned Python source,
dependency-free score recomputation, focused numeric tests, and a visual report.
Source compilation, local saved-CSV evaluation, and final fusion reconstruction
from the retained intermediate tables are checked. The latter produced the exact
saved final CSV SHA256
`b003f34a9d8936845a20421c4dc98dbfd9ce5c35c39fccccfcfedcb9bfaff49c`.
This validates configuration selection and fusion, not new model inference.
Fresh image-to-
prediction inference on a clean machine has not been revalidated here.

## Data Layout

Prepare authorized competition assets locally; do not commit these directories:

```text
data/id_map.csv          combo, combo_name, id, food, container, split
data/work/images/       combo_01.JPG ... combo_14.JPG
data/_HELD_OUT/          isolated GT assets, post-hoc analysis only
cache/                  masks and intermediate model outputs
outputs/                model-derived CSVs and geometry
submissions/            generated prediction CSVs
external/               separately obtained third-party source
```

`src/paths.py` resolves these paths relative to the repository root. Do not place
GT assets inside `data/work/`. The ID mapping uses IDs 1-10 for the public subset
and 11-34 for private evaluation; repeated food observations can share an ID.

## Environments

The source includes two historically separate GPU environments because mesh
generation and the main Transformers stack used different versions.
The following versions were observed on the retained local environments during
export, not reconstructed from a training-time lock file:

| Package | Main | Mesh generation |
| --- | --- | --- |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu128 |
| Transformers | 5.15.1 | 4.46.0 |
| NumPy | 2.2.6 | 1.26.4 |
| trimesh | 5.0.0 | 5.0.0 |
| accelerate | 1.14.0 | not audited |
| kernels | 0.16.0 | not audited |

Model access approval may be required. Obtain Hunyuan3D 2.1 separately under
`external/Hunyuan3D-2.1/`; `gen3d.py` imports its `hy3dshape` package. Optional
MoGe diagnostics require the separately installed MoGe implementation.
Do not assume `requirements-core.txt` is a complete GPU installation recipe.

## Historical Final Configuration

Main components: SAM 3 mask refinement, Depth Pro, Hunyuan3D 2.1,
`Qwen/Qwen3.8-27B-FP8` as named in the archived source, crop expansion 5.0,
mesh alignment surface sampling seed 0, VLM weight multiplier 0.25, shrinkage off.
Model availability and compatibility must be verified before a new GPU run.

Stage order after preparing metadata and model environments:

1. Segmentation/refinement: `src/segment_merge.py`, `src/segment_refs.py`.
2. Plate and visual priors: `src/qwen_qc.py`, `src/qwen_shape.py`.
3. Mesh generation in the separate environment: `src/gen3d.py`.
4. Alignment: `src/align_mesh.py --backend depthpro --expand 5 --mesh-seed 0`.
5. Shape tables: `src/apply_shape.py --backend depthpro --expand 5 --out
   shape_depthpro_e5.0.csv` and the corresponding `--rim --out
   shape_depthpro_e5_rim.csv` run.
6. Final fusion: `python scripts/build_final.py`.

These stages require generated masks, model outputs, and metadata, not provided
in the public export. They are an execution map, not a guaranteed one-command
reproduction. Run each stage under its appropriate environment.

`build_final.py` fixes the historical e5 input globs explicitly and rejects
missing tables or IDs before writing predictions. Container volumes override
the fusion path. Keep GT-based scoring separate from prediction generation.

## Randomness and Limitations

The saved final CSV and the mesh-alignment sampling fix have been checked.
Other generation/diagnostic code still contains stochastic sampling; whole-
pipeline bitwise determinism across machines is not established. The archived
GT-access harness is advisory, not an OS-level sandbox. The project owner
clarified that the GT comparisons were post-hoc checks. The retained archive
does not independently establish the configuration-selection chronology.

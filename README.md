# fractal-texture-defect-detection

Training-free detection of defects in textures by invariant block matching against a few defect-free reference images.
The detector is derived from the encoder of a fractal (partitioned iterated function system, PIFS) image codec: every block of
a test image is matched against all blocks of the reference in the eight rotations and reflections of the square, with a bounded
least-squares fit of contrast and brightness; the normalised residual of the best match is the anomaly score. It needs no
training, no neural network, no GPU and no pretrained weights.

This repository contains the code, the protocol and the per-tile scores behind the paper

> V. A. Kalinkin, *From fractal image coding to training-free texture defect detection: invariant block matching with a few
> reference images*. Manuscript for Electronic Letters on Computer Vision and Image Analysis (ELCVIA), 2026.

## Main results

Image-level AUROC on six woven fabrics of AITEX (fabric 05 excluded by a rule fixed in advance), design v3, mean over fabrics.
Intervals are 95% cluster-bootstrap intervals over fabric strips.

| Method                              | 1 reference | 4 references |
|-------------------------------------|:-----------:|:------------:|
| Plain pixel matching                | 0.819       | 0.819        |
| Light variant (bounded fit)         | 0.917       | 0.917        |
| **Ours (full)**                     | **0.915**   | **0.910**    |
| PatchCore A (all features)          | 0.895       | 0.891        |
| PatchCore B (10% coreset)           | 0.889       | 0.899        |
| PaDiM                               | n/a         | 0.866        |

* Ours minus the better PatchCore configuration: +0.020 [-0.016, +0.056] (1 reference), +0.011 [-0.034, +0.049] (4 references).
  No difference is detected, and differences of up to about 0.05 are not excluded.
* Ours minus plain matching: +0.096 [+0.057, +0.138] and +0.091 [+0.056, +0.131]. Almost all of this comes from centring the
  blocks (zero-mean matching alone: 0.913); the contrast fit with bounds, the isometries and the local-brightness term add no
  detectable gain on these fabrics.
* In a control design (defect-free tiles taken from the defective strips) all methods lose 0.08 to 0.17 AUROC and PatchCore is
  ahead by 0.029, with an interval that includes zero.
* PatchCore is better in pixel-level localisation, on irregular MVTec AD textures and on three of the six fabrics.
* Run time on a laptop CPU (indicative): about 0.05 s per tile for the light variant, 0.53 s for the full method, 0.19 s for
  PatchCore including training.
* Not compared: detectors built on foundation features (for example AnomalyDINO-S). The comparison is with CNN-feature memory
  banks of one library, not with the current state of the art.

The protocol, its two amendments and the outcome of the decision rule are in [PROTOCOL.md](PROTOCOL.md).

## Contents

| File | Purpose |
|------|---------|
| `eval_mvtec.py` | The detector (`plain`, `affine`, `affine_iso`) and its evaluation on a dataset in MVTec AD folder layout |
| `prepare_aitex_v3.py` | AITEX to MVTec layout, design v3: main design and control design (`--variant ctrl`), writes `manifest.json` |
| `prepare_aitex.py` | Earlier designs v1 and v2 (kept for the history table of the paper) |
| `measure_light.py` | Illumination measurements on AITEX (between strips, along the strip) |
| `filter_edges.py` | Removes background tiles from saved scores (design v2) |
| `make_shifted.py` | Copies of MVTec AD textures under illumination shifts and rotation, so that all methods see the same files |
| `run_patchcore.py` | PatchCore and PaDiM through anomalib; saves per-tile scores |
| `bootstrap_ci.py` | Paired bootstrap for AUROC and FPR95; `--by-strip` resamples whole strips (cluster bootstrap) |
| `summarize_v3.py` | Tables from the saved scores, decision rule |
| `run_all_v3.sh` | The batch of runs behind the paper: data preparation, main design with 1 and 4 references, control design, ablation variants (sections 1-4; see the note below) |
| `make_report_v3.sh`, `optional_intervals.sh` | Reports and intervals recomputed from saved scores in seconds |
| `v3res/` | Per-tile scores of every run, `REPORT_core.txt`, `intervals.txt` |
| `PROTOCOL.md` | Protocol fixed before the final run, amendments, deviation log |

Comments and some log messages in the scripts are in Russian; the command-line options are in English.

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt             # our method and the statistics (numpy, Pillow)
pip install -r requirements-baselines.txt   # additionally PatchCore and PaDiM (anomalib 2.6.2, PyTorch)
```

## Data

The datasets are not redistributed here.

* **AITEX Fabric Image Database** (Silvestre-Blanes et al., Autex Research Journal, 2019): <https://www.aitex.es/afid/>.
  The scripts expect the folders `Defect_images`, `Mask_images` and `NODefect_images/<fabric>` (as in the Kaggle copy).
  The Kaggle copy is licensed CC BY-NC-ND 4.0 (non-commercial, no derivatives), which is why this repository publishes
  scripts and scores but no tiles or other adapted images. Cite the original paper.
* **MVTec AD** (Bergmann et al., CVPR 2019): <https://www.mvtec.com/company/research/datasets/mvtec-ad>, licence CC BY-NC-SA 4.0
  (non-commercial use).

## Reproducing the paper

### A. Recompute the statistics from the saved scores (seconds, no models)

`v3res/` holds the score of every test tile for every method and setting, so all intervals and tables can be recomputed
without the data and without rerunning any model.

```bash
python3 summarize_v3.py --res v3res                   # tables, operating points, decision rule
bash make_report_v3.sh                                # intervals against PatchCore and PaDiM -> v3res/REPORT_core.txt
bash optional_intervals.sh > intervals.txt            # intervals against plain matching and for the ablation

# one comparison by hand: ours (full) against PatchCore A, one reference, cluster bootstrap over strips
python3 bootstrap_ci.py --by-strip --exclude fabric_05 \
    --a v3res/ours_s0_1.json --a-method affine_iso --a-shots 1 \
    --b v3res/pcA_s0_1.json  --b-method patchcore_A
# add --metric fpr95 for the false-alarm rate at 95% detection
```

### B. Rerun everything (laptop CPU)

Set `ARCH` (the AITEX `archive` folder) and `DATA` (where the prepared tiles go) at the top of the scripts, then run them
from the repository root.

```bash
bash run_all_v3.sh       # several hours on a laptop CPU; writes the score files to v3res/ and the report v3res/REPORT.txt
```

The score files in `v3res/` were produced by sections 1 to 4 of `run_all_v3.sh`. The run was stopped after the ablation section: the step `abl_iso` (isometries without the brightness term) did not complete, and sections 5 to 7 (sensitivity to parameters, gain invariance, choice of the reference) were not run, so their outputs are not in `v3res/` and are not used in the paper. `make_report_v3.sh` then built `v3res/REPORT_core.txt` from the saved scores, and `optional_intervals.sh` built `intervals.txt`.

`caffeinate -i bash ...` keeps a Mac awake. PatchCore results depend on the anomalib and PyTorch versions and on the random
coreset, so small differences from the numbers above are expected; `v3res/` holds the scores behind the paper.

## Using the detector on your own images

Arrange the images in the MVTec AD layout

```
<root>/<category>/train/good/*.png          the references (the first --shots images in sorted order are used)
<root>/<category>/test/<type>/*.png         test images; <type> = good for defect-free ones
<root>/<category>/ground_truth/<type>/<name>_mask.png     masks of the defective ones
```

and run the full method with the parameters of the paper:

```bash
python3 eval_mvtec.py --root <root> --cat <category> --shots 1 \
    --methods affine_iso --norm --mean-w 0.1 --block 16 --stride 4 --smin 0.6 --smax 1.7 \
    --save-scores scores.json
```

| Variant | Options |
|---------|---------|
| Plain pixel matching | `--methods plain` |
| Centring only (contrast fixed to 1) | `--methods affine --smin 1 --smax 1 --mean-w 0` |
| Light (bounded fit, no isometries, no brightness term) | `--methods affine --norm --mean-w 0` |
| Full | `--methods affine_iso --norm --mean-w 0.1` |

`--smin` and `--smax` are the contrast bounds (`--smax 0` removes the upper bound), `--norm` divides the residual by the squared
contrast, `--mean-w` is the weight of the local-brightness term, `--block` and `--stride` are the block size and the stride of
the reference bank. Images are resized to 256 x 256 grey scale (`--size`). The image score is the maximum of the defect map;
the per-tile scores are written to the file given by `--save-scores`.

## Limitations

* AITEX is one fabric database with laboratory lighting; the illumination shifts on MVTec AD are synthetic. Design decisions
  were made on MVTec AD, so the MVTec AD results are not confirmatory.
* The one-reference and four-reference settings are not evaluated on identical test sets (a strip used as a reference leaves the
  test set); methods are compared within a setting, on identical tiles.
* The intervals describe one choice of the reference per fabric. The intervals for plain matching and for the ablation were
  computed after the main results were known and are not part of the decision rule (see the deviation log in the paper).
* Timings are indicative: one laptop that was not dedicated to the runs.

## Citation

If you use the code or the scores, please cite the paper (see `CITATION.cff`). Please also cite the datasets and, for the
baselines, PatchCore (Roth et al., CVPR 2022) and anomalib (Akcay et al., ICIP 2022).

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets have their own terms, see above.

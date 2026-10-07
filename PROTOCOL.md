# Experiment protocol (fixed BEFORE the final results were obtained)

(Translated from the Russian original. The amendments below record what was found and changed after earlier runs; all
versions of the test design are reported in the paper.)

**Hypothesis.** On periodic (woven, grid-like) textures with 1-5 reference images, invariant block matching with a
local-brightness correction ("our method") detects defects at least as well as PatchCore, and on part of the data better.

**What counts as our method (parameters are not tuned on test data).**
Block 16, bank stride 4, 8 isometries, contrast s in [0.6, 1.7], error normalised by s^2, local-brightness correction with
weight 0.1 (window of 15 blocks), image score = maximum over the map, images resized to 256 x 256, grey scale.

**Competitor: PatchCore (anomalib).** Run in several configurations; the table shows the one that is BEST for it by mean
image-level AUROC, so that the competitor cannot be said to be weakened:
  A) coreset 1.0, default backbone and layers;  B) coreset 0.1 (the default value);  C) layers layer1 layer2 layer3.
The comparison is on the same files (prepared by make_shifted.py).

**Data.** MVTec AD, texture classes: grid, carpet (periodic), leather, tile, wood (for completeness);
AITEX: fabrics (each fabric separately; 1 and 4 reference tiles of 256 x 256).

**AITEX preparation (prepare_aitex.py).** Strips of 4096 x 256 are cut into tiles of 256 x 256 at the original resolution.
References: k tiles spread evenly along the length of the FIRST defect-free strip of the fabric (k = 1 and 4). Test: every
4th tile of the remaining defect-free strips, plus the tiles of defective strips whose mask has at least 20 defect pixels.
Both methods work on the same files. Fabrics with fewer than 5 defective test tiles are shown in the table but are not part
of the decision rule (AUROC on them is unreliable).

**Metrics.** Image-level AUROC, pixel-level AUROC. Time per image is measured separately, with sleep mode disabled
(caffeinate).

**Decision rule (before looking at the AITEX results).**
- The variant "paper on periodic textures" is chosen if on AITEX our method has a mean image-level AUROC not lower than
  the best PatchCore configuration, with 1 and with 4 references, AND wins on at least half of the fabrics (among those
  counted).
- Otherwise we write an honest comparative study (variant 1) for a more modest venue, or stop.

**What we do not do.** We do not tune weights, thresholds or shifts on test images. We do not look for a "hard" lighting
shift until the competitor breaks. The results of every run are kept in full, including the unsuccessful ones.

**Amendment 1 (made after the first AITEX run, before the second).**
Inspection of the heat maps showed that the end tiles of the strips contain background outside the fabric (for example
fabric 03: about a quarter of the defect-free test tiles). The reference is taken from the middle of the strip, so the border
between fabric and background looks like a defect to any method that compares with the reference. This is an error of data
preparation, not a property of the methods. Fix: tile positions where, on the defect-free strips of the given fabric, the
share of background pixels (brightness >= 250 or <= 5) is on average above 1 % are excluded from references and test. The
1 % threshold was chosen before recomputing and is not tuned. The fix is applied identically to both methods. With 1
reference the reference tile (8) contains no edges. With 4 references (tiles 3, 6, 9, 12), for fabrics 00 and 03 tile 3 is
entirely background and tile 4 partly (the edge positions of fabrics 00 and 03 are 0-4): there are effectively 3 fabric
references plus background, identically for both methods. The models are not rerun: edge test tiles are removed from the
saved scores (filter_edges.py) and the metrics are recomputed. Edge positions found: 00: 0-4; 01: 0-1; 02: 0; 03: 0-4;
04: 2; 05: 0.

**Outcome of the decision rule (after Amendment 1, AITEX without fabric 05).**
1 reference: ours 0.922 against the best PatchCore (A) 0.881; difference +0.042, 95 % CI [+0.001, +0.083]; wins 4 of 6.
4 references: ours 0.906 against the best PatchCore (B) 0.895; difference +0.011, 95 % CI [-0.029, +0.051]; wins 2 of 6
(1 tie). The condition "not lower on average" is met in both settings; the condition "wins on at least half of the
fabrics" with 4 references is NOT met. Therefore, by the protocol, the work is written as a comparative study (variant 1),
not as a claim of superiority on periodic textures. The results of the first run are kept and mentioned in the paper as a
result with a data-preparation error. The decision rule and the method parameters are not changed.

**Amendment 2 (after an external review of the draft; design v3).**
The review pointed to biases of the AITEX test sample that remained after Amendment 1: (a) defect-free tiles were taken
only from positions 0/4/8/12 and the reference sat at position 8, whereas defective tiles came from all positions; (b) with
4 references all of them came from one strip, and for fabrics 00 and 03 one of them was background; (c) defect-free and
defective tiles were taken from different strips. Illumination measurements confirmed this: brightness along a strip changes
by 5-56 % (fabric 06: from 87 to 158), and the mean brightness of the defective strips of fabrics 00 and 01 varies several
times more than that of the defect-free ones (CV 14 % against 0.5-3.6 %).
The new design (prepare_aitex_v3.py): positions with background are excluded; references from k DIFFERENT strips at the
middle allowed position; defect-free tiles from ALL allowed positions in equal numbers; a control variant in which the
defect-free tiles are taken from defect-free parts of defective strips (distance to a labelled defect greater than one
position). The method parameters, the decision rule and the list of compared methods did not change; PaDiM and the metric
"false-alarm rate at 95 % detection" were added. The models were rerun. The decision was taken AFTER looking at the results
of the previous versions, so the results of all versions are kept and reported. Fabric 05 remains excluded (1 defective
tile).

**Outcome of the decision rule under design v3 (6 fabrics, image-level AUROC).**
Main design, 1 reference: ours 0.915, best PatchCore (A) 0.895, difference +0.020 [-0.016, +0.056]; wins 3 of 6.
Main design, 4 references: ours 0.910, best PatchCore (B) 0.899, difference +0.011 [-0.034, +0.049]; wins 3 of 6.
Conditions (i) "not lower on average" and (ii) "wins at least half" are MET in both settings of the main design.
Control, 1 reference: ours 0.781, PatchCore (A) 0.811, difference -0.029 [-0.103, +0.023]; condition (i) is NOT met.
The rule was set as a NECESSARY condition for a claim of superiority. It is met on the main design but not on the control,
all intervals include zero, and the design was changed after the results were seen. Therefore no claim of superiority is
made; the work is written as a comparison: "no difference detected; differences of up to about 0.05 are not excluded".
Limitation of the result: PaDiM with 1 reference gives degenerate tile scores (a constant); those rows are not used.

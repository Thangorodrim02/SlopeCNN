# data

The SlopeCNN v3 dataset: 346 Rocscience Slide2 models of Section 4 of the Çöllolar
open-pit lignite mine (Elbistan, Türkiye), with their soil properties and simplified
Bishop factor-of-safety (FoS) labels.

```
data/
├── slide2_models/                              Feed<N>.slim, the 346 Slide2 models (N = geometry ID)
├── dataset_v3_346_geometries_FoS.xlsx          main workbook (start here)
├── soil_properties_346geom_v3_cnn_input.xlsx   the same data, one header row (read by the code)
├── slide2_validation/                          the 50 models used to check the labels against Slide2
└── scripts/                                    how the material values were sampled and written
```

The FoS labels and critical circles are also in `../slopelem/data/labels_v3_346.csv`.

## The models

* **One cross-section.** All 346 models are variations of Section 4 (Öge, 2008): the
  same six layers, from top to bottom loam, blue clay, gyttja, lignite, black clay and
  green clay, and a 150 m high benched slope. Loam is absent in 27 models and blue clay
  in 2, as in the original drawings.
* **Geometry.** The overall slope angle is 5–25°. The five layer boundaries are straight
  lines whose dips vary within ±3° of the section dips. The models were built by hand in
  Slide2, and the water table was drawn by hand in each model.
* **Soil properties.** Six properties per layer: saturated and unsaturated unit weight
  (γ_sat, γ_unsat), peak and residual cohesion (c′, c_r) and peak and residual friction
  angle (φ′, φ_r). Each was drawn uniformly within ±20 % of its base value (seed 42), and
  a pair was redrawn until γ_sat ≥ γ_unsat and residual ≤ peak. Base values: laboratory
  peak and residual strengths from Öge (2008), unit weights from Özbay & Cabalar (2015);
  they are in the sheet *Reference Values*.
* **Slide2 materials.** Anisotropic strength: Cohesion 1 / Phi 1 (along the bedding) =
  residual, Cohesion 2 / Phi 2 (across the bedding) = peak, anisotropy angle = layer dip
  (sheet *Slide2 angles*). Unit weight γ_unsat above the water table and γ_sat below it.
  Pore pressure u = Hu · γ_w · depth below the water table, with Hu from Öge (2008):
  loam 0.36, blue clay 0.31, gyttja 0.78, lignite 1.00, black clay 0.87, green clay 0.47.
* **IDs.** The geometry ID N is the N of `FeedN.slim`. IDs run from 1 to 494 and are not
  consecutive.

## FoS labels

Simplified Bishop method on circular slip surfaces, calculated with
[`slopelem`](../slopelem) directly on each Slide2 model (200 slices; lowest FoS of two or
three global searches and two small-circle searches; Slide2's admissibility rules
m_α ≥ 0.2 and no reverse curvature). Range 0.719–2.598, median 1.223; 88 models below
1.0, 106 between 1.0 and 1.3, 152 at 1.3 or above.

Every label is exactly reproducible from its critical circle:

```bash
cd slopelem
python examples/reproduce_labels.py ../data/slide2_models      # 346 of 346 labels reproduced exactly
```

## Main workbook: `dataset_v3_346_geometries_FoS.xlsx`

| Sheet | Contents |
|---|---|
| Read me | how the data were made |
| 346 Geometries — CNN Dataset | one row per model: FoS, slope angle, dip and deviation of each boundary, 6 properties × 6 layers |
| Statistics | minimum, maximum, mean and standard deviation of every property |
| Reference Values | base values, Hu, normalisation maxima of the CNN |
| Slide2 angles | anisotropy angle of each layer in each model |
| FoS circles | critical circle of each model (centre, radius, entry and exit x, half-angle β) |

`soil_properties_346geom_v3_cnn_input.xlsx` holds the same values with the column names
used by `slopecnn/scripts/tabular_baselines.py` (`geometry_id`, `Factor of Safety (FS)`,
`slope_angle_deg`, `bndN_delta_deg`, `layerN_<Layer>_<property>`).

## Validation against Slide2: `slide2_validation/`

50 models were run in Slide2 with Slide2's own search and with the slopelem critical
circle added as a single surface: 20 chosen at random (5 per FoS range) and 30 targeted
ones (small crest circles, FoS closest to 1.0, shallowest slopes).

| File | Contents |
|---|---|
| `slide2_computed_20_models.zip` | the 20 random models, saved in Slide2 with their results |
| `slide2_computed_targeted_30_models.zip` | the 30 targeted models, saved in Slide2 with their results |
| `slide2_validation_results.csv`, `slide2_validation_targeted_results.csv` | Slide2 and slopelem FoS of each model |

Mean |difference| 0.055 %, largest 0.147 %, same critical circle in 44 of 50. The full
table is `../slopelem/validation/Slide2_validation_50_models.xlsx`; to recompute it,
unzip both archives into one folder and run
`python examples/validate_against_slide2.py <folder>` in `../slopelem`.

## `scripts/`

`build_dataset.py` and `slimtool.py` drew the soil properties and wrote them into the
Slide2 models. They need the hand-drawn models without materials and two working
workbooks, which are not part of the repository; they are included to document the
sampling. The models in `slide2_models/` are their output.

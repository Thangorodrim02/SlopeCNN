# slopelem

Simplified Bishop limit-equilibrium analysis of Rocscience Slide2 slope models, with
anisotropic (bedding-dependent) Mohr–Coulomb strength and Hu pore pressures.

`slopelem` is the LEM model generator of the SlopeCNN repository: it computed the
factor-of-safety (FoS) labels of the **SlopeCNN v3 dataset** (346 Slide2 models,
`../data/slide2_models`). It reads the Slide2 model files (`.slim`) directly, so the
labels come from exactly the same geometry, materials and water table as the Slide2
models. It was validated against Slide2 on 50 of those models (see [Validation](#validation)).

```
slopelem/
├── pyproject.toml
├── README.md
├── src/slopelem/
│   ├── model.py           Slide2 file reader: geometry, materials, water table, slice columns
│   ├── bishop.py          simplified Bishop FoS of one circle + Slide2 admissibility rules
│   ├── search.py          critical-circle search (global + small-circle, fixed seeds)
│   ├── slide2_results.py  reader for Slide2 results stored in a computed .slim (.s01)
│   └── cli.py             command-line interface
├── data/labels_v3_346.csv FoS label and critical circle of every model
├── examples/              compute_one.py, reproduce_labels.py, validate_against_slide2.py
├── validation/            Slide2 validation workbook (50 models) and its CSV
└── tests/                 pytest suite + small set of Slide2 models
```

## Installation

Python ≥ 3.9, NumPy and SciPy.

```bash
cd slopelem
pip install -e ".[test]"
pytest                 # 10 fast tests, ~1 s
pytest -m slow         # full critical-circle search on one model, ~1 min
```

## Usage

### Python

```python
from slopelem import SlopeModel, find_critical_circle, bishop_fos, circle_through_ground_points

model = SlopeModel.from_file("Feed7.slim")

# 1. critical circle with the label protocol
res = find_critical_circle(model)
print(res.fos, res.found_by, res.circle)

# 2. FoS of a given circle (through ground points x_entry, x_exit, half-angle beta in rad)
c = circle_through_ground_points(model, 202.91, 279.31, 0.598611071)
print(bishop_fos(model, c).fos)   # 1.226 = label of Feed7
```

`bishop_fos` returns a `BishopResult` with the FoS, the slice-by-slice data
(`weight`, `alpha`, `width`, `cohesion`, `phi`, `pore_pressure`, `material`),
the minimum m_α and a convergence flag. An inadmissible surface gives `fos = nan`.

### Command line

```bash
slopelem fos    Feed7.slim                         # critical circle of one model
slopelem circle Feed7.slim 261.34 1217.56 72.08    # FoS of a circle (xc, yc, R)
slopelem batch  Models/ --out fos.csv --workers 4  # every .slim in a folder
```

## Method

### Model input

Everything is read from the Slide2 `.sli` file inside the `.slim` archive:

| Item | Slide2 source |
|---|---|
| Geometry | triangulated material cells (`cells`, `vertices`) |
| Ground surface | top of the material cells, sampled every 0.5 m |
| Slope limits (search range) | `slope limits` |
| Water table | `water table` polyline |
| Materials 1–6 | `soilK` lines: `c, phi, c2, phi2, ang, uw, uwbwt, water: Hu` |

Units: m, kN/m³, kPa, degrees. The slope faces right (crest left, toe right), as in
all SlopeCNN models. γ_w = 9.81 kN/m³.

### Anisotropic strength

Slide2's *Anisotropic strength* model. With δ the angle between the slice base and
the material's anisotropy direction (`ang`, counter-clockwise from +x):

```
c(δ)   = c1 · cos²δ + c2 · sin²δ
φ(δ)   = φ1 · cos²δ + φ2 · sin²δ
```

In the SlopeCNN models slot 1 holds the **residual** strength (along bedding) and
slot 2 the **peak** strength (across bedding), and `ang` is the dip of each layer.

### Pore pressure and unit weight

```
u = Hu · γ_w · h_w
```

where h_w is the vertical depth of the slice base midpoint below the water table
(u = 0 above it). Soil above the water table has unit weight `uw` (γ_unsat), below it
`uwbwt` (γ_sat).

### Simplified Bishop

The sliding mass is divided into 200 equal-width vertical slices; each slice weight
is integrated over 3 sub-columns. With α the base inclination, b the width, W the
weight, and c, φ, u at the base midpoint:

```
FoS = Σ [ (c·b + (W − u·b)·tanφ) / m_α ]  /  Σ W·sinα
m_α = cosα · (1 + tanα · tanφ / FoS)
```

solved by fixed-point iteration (tolerance 1e-7).

### Slide2 admissibility rules

Two Slide2 rules are reproduced, because they decide which surfaces Slide2 accepts:

| Slide2 code | Rule | Effect |
|---|---|---|
| −112 | m_α < 0.2 on any slice | surface rejected |
| −114 | reverse curvature: centre below the higher of the two ground intersections | surface rejected |

A circle must also stay below the ground surface between its entry and exit points.

### Critical-circle search

A trial circle is parameterised by its ground intersections and half-angle,
`p = (x_entry, x_exit, β)` with `R = chord / (2 sin β)`, so every trial circle cuts
the ground exactly at both ends. Inadmissible circles get a penalty FoS of 10⁶.

| Stage | Sampling | Refinement |
|---|---|---|
| Global search, seeds 1 and 2 | 4 000 random (x_entry, x_exit, β) in the slope limits | Nelder–Mead on the best 12 |
| Extra global search, seed 3 (only if seeds 1 and 2 differ by > 1 %) | 6 000 random | best 20 |
| Small-circle search, seeds 11 and 12 | 2 000 random chords of 15–200 m | best 6 |

The label is the lowest FoS of all runs. All settings are in `SearchSettings`;
the defaults are the label protocol, so a run is fully deterministic.

## The dataset labels

`data/labels_v3_346.csv` holds, for every model `FeedN.slim`:

| Column | Meaning |
|---|---|
| `geometry_id` | N |
| `FoS` | label (3 decimals) |
| `found_by` | `global search` (272 models) or `small-circle search` (74) |
| `xc, yc, radius` | critical circle (m) |
| `x_entry, x_exit, beta_rad` | the same circle as search parameters |

Every label is **exactly reproducible** from its published circle:
`round(bishop_fos(model, circle_through_ground_points(model, x_entry, x_exit, beta_rad)).fos, 3)`
equals `FoS` for all 346 models (`examples/reproduce_labels.py`). Labels range from
0.719 to 2.598 (median 1.223).

## Validation

1. **Slice level.** On Slide2's own critical circle of a computed model, with Slide2's
   slice boundaries, the base material, cohesion and friction angle of every slice
   match Slide2's exported slice data exactly and the FoS within 0.1 %
   (`test_slice_data_match_slide2`).
2. **50 dataset models.** Each was run in Slide2 with its own search and with the
   solver's critical circle added as a single surface
   (`validation/Slide2_validation_50_models.xlsx`):

| Group | Models | Mean diff. (%) | Mean \|diff.\| (%) | Max \|diff.\| (%) | Same critical circle |
|---|---|---|---|---|---|
| Random, 5 per FoS range | 20 | −0.021 | 0.062 | 0.147 | 17 of 20 |
| Small crest circles | 12 | −0.058 | 0.058 | 0.119 | 12 of 12 |
| FoS closest to 1.0 | 10 | −0.000 | 0.054 | 0.100 | 7 of 10 |
| Shallowest slopes | 8 | −0.031 | 0.037 | 0.066 | 8 of 8 |
| **All** | **50** | **−0.027** | **0.055** | **0.147** | **44 of 50** |

Difference = (label / Slide2 FoS − 1) × 100. `examples/validate_against_slide2.py` recomputes this table from the computed Slide2 files in `../data/slide2_validation/` (unzip both archives into one folder). The script reads Slide2's FoS at full precision, so it prints 0.056 % and 0.148 % instead of the 0.055 % and 0.147 % of the table, which uses Slide2's FoS rounded to 4 decimals. In the 6 cases where Slide2's search
ended on a different circle, the two minima differ by less than 0.15 %.

## Limitations

* Circular surfaces and the simplified Bishop method only.
* Built for the SlopeCNN model family: six materials, one water table, slope facing
  right. Other Slide2 features (supports, loads, tension cracks, non-circular
  surfaces, other strength models) are not read.
* The search is stochastic with fixed seeds; it finds the same minimum as Slide2's
  search to within 0.15 % on the validation set, but is not guaranteed to find the
  global minimum on arbitrary geometries.

## Licence

MIT, see [`LICENSE`](../LICENSE) in the repository root.

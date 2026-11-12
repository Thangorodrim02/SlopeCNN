"""Build the 346-geometry dataset (v3): one cross-section, redrawn properties,
Excel workbook + Slide2 models with anisotropic materials.
usage: python build_dataset.py <template_v2.xlsx> <hsu_500.xlsx> <pool_dir> <out_dir>"""
import os, re, sys, copy
import numpy as np, pandas as pd, openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter as CL
import slimtool as t

TPL, HSU, POOL, OUT = sys.argv[1:5]
SEED, PCT = 42, 0.20
LAYERS = ["Green Clay", "Black Clay", "Lignite", "Gyttja", "Blue Clay", "Loam"]
PROPS = ["γ_sat", "γ_unsat", "c_peak", "c_res", "φ_peak", "φ_res"]
UNITS = ["kN/m³", "kN/m³", "kPa", "kPa", "°", "°"]
HU = {"Loam": 0.36, "Blue Clay": 0.31, "Gyttja": 0.78, "Lignite": 1.00, "Black Clay": 0.87, "Green Clay": 0.47}
C0, FIRST = 14, 5
DS = "346 Geometries — CNN Dataset"

pool = {int(re.search(r"Feed(\d+)\(", f).group(1)): f for f in os.listdir(POOL)}
ids = sorted(pool)
assert len(ids) == 346
LAST = FIRST + len(ids) - 1
h = pd.read_excel(HSU).set_index("geometry_id").loc[ids]

wb = openpyxl.load_workbook(TPL)
ws = wb["300 Geometries — CNN Dataset"]; ws.title = DS
ref = wb["Reference Values"]
base = {ref.cell(r, 1).value: [float(ref.cell(r, c).value) for c in range(2, 8)] for r in range(4, 10)}

# ---- draw properties (uniform ±20 %, pairwise redraw until sat>=unsat, res<=peak)
rng = np.random.default_rng(SEED)
def pair(hi_b, lo_b):
    while True:
        hi = hi_b * rng.uniform(1 - PCT, 1 + PCT); lo = lo_b * rng.uniform(1 - PCT, 1 + PCT)
        if hi >= lo:
            return round(hi, 4), round(lo, 4)
vals = {}
for gid in ids:
    for lay in LAYERS:
        b = base[lay]; v = [None] * 6
        for hi_i, lo_i in [(0, 1), (2, 3), (4, 5)]:
            v[hi_i], v[lo_i] = pair(b[hi_i], b[lo_i])
        vals[(gid, lay)] = v

# ---- data sheet
style_row = [copy.copy(ws.cell(FIRST, c)._style) for c in range(1, 50)]
for r in range(FIRST, max(ws.max_row, LAST) + 1):
    for c in range(1, 50):
        ws.cell(r, c).value = None
for i, gid in enumerate(ids):
    r = FIRST + i; row = h.loc[gid]
    for c in range(1, 50):
        ws.cell(r, c)._style = copy.copy(style_row[c - 1])
    ws.cell(r, 1).value = gid
    ws.cell(r, 3).value = float(row["slope_angle_deg"])
    for b in range(1, 6):
        ws.cell(r, 2 + 2 * b).value = float(row[f"bnd{b}_angle_deg"])
        ws.cell(r, 3 + 2 * b).value = float(row[f"bnd{b}_delta_deg"])
    for k, lay in enumerate(LAYERS):
        for j in range(6):
            ws.cell(r, C0 + 6 * k + j).value = vals[(gid, lay)][j]
ws.cell(1, 1).value = ("Elbistan — Slope Stability CNN Dataset v3  |  346 Geometries, one cross-section × "
                       "Redrawn Soil Properties (uniform ±20 %, seed = 42, γ_sat ≥ γ_unsat, residual ≤ peak)")
ws.cell(4, 2).value = "Factor of Safety (FS) — pending Slide2 rerun"

# ---- statistics (live formulas)
st = wb["Statistics"]
st.cell(1, 1).value = "Özellik İstatistikleri — 346 Geometri — v3"
r = 3
for k, lay in enumerate(LAYERS):
    for j, p in enumerate(PROPS):
        rg = f"'{DS}'!${CL(C0+6*k+j)}${FIRST}:${CL(C0+6*k+j)}${LAST}"
        st.cell(r, 4).value = f"=ROUND(MIN({rg}),2)"; st.cell(r, 5).value = f"=ROUND(MAX({rg}),2)"
        st.cell(r, 6).value = f"=ROUND(AVERAGE({rg}),2)"; st.cell(r, 7).value = f"=ROUND(STDEV({rg}),2)"
        r += 1

# ---- reference: Hu only (drop the old-model column)
ref.cell(2, 9).value = None
for rr in range(4, 10):
    ref.cell(rr, 9).value = None

# ---- remove v2-only sheets, rebuild notes
for name in ["Changes v2", "Old FoS (not valid)"]:
    del wb[name]
bold = Font(bold=True); hdr = PatternFill("solid", fgColor="D9E1F2")
nt = wb.create_sheet("Read me", 0)
notes = [
    "Dataset v3 — 346 hand-made Slide2 models of ONE cross-section (the section in config.py)",
    "Geometry: IDs 1–300 from 'CNN Feed', IDs 301–494 from selected_slim.zip; geometry values from soil_properties_v2_500geom.xlsx (all checked against the drawn models).",
    "Dropped: the 100 Section-4 models (old IDs 501–600) — their geometry differs from the CNN template.",
    "Properties: all 6 properties of all 6 layers drawn uniformly within ±20 % of base (Reference Values); pairs redrawn until γ_sat ≥ γ_unsat and residual ≤ peak; seed 42.",
    "Slide2 materials: Anisotropic strength. Cohesion 1/Phi 1 (along bedding) = residual; Cohesion 2/Phi 2 = peak; Angle (ccw to 1) = layer dip (sheet 'Slide2 angles').",
    "Unit weight above water table = γ_unsat; below water table = γ_sat; Hu = Öge (2008) Table 4.4 (Reference Values col H).",
    "FoS column is empty until the Slide2 rerun. After the rerun, select the final 300 from these 346 using the new FoS.",
]
for i, s in enumerate(notes, 1):
    nt.cell(i, 1).value = s
nt.cell(1, 1).font = Font(bold=True, size=12)

ag = wb.create_sheet("Slide2 angles")
heads = ["Geometry ID"] + [f"{L} angle (ccw to 1) (°)" for L in ["Loam", "Blue Clay", "Gyttja", "Lignite", "Black Clay", "Green Clay"]] + ["Source model file"]
for j, s in enumerate(heads, 1):
    c = ag.cell(1, j); c.value = s; c.font = bold; c.fill = hdr
for i, gid in enumerate(ids):
    rw = 2 + i
    xr = {f"bnd{b} Angle (°)": h.loc[gid, f"bnd{b}_angle_deg"] for b in range(1, 6)}
    ag.cell(rw, 1).value = gid
    for j, L in enumerate(["Loam", "Blue Clay", "Gyttja", "Lignite", "Black Clay", "Green Clay"]):
        ag.cell(rw, 2 + j).value = round(t.layer_dip(xr, L), 4)
    ag.cell(rw, 8).value = pool[gid]
ag.column_dimensions["A"].width = 12; ag.column_dimensions["H"].width = 24
for c in "BCDEFG":
    ag.column_dimensions[c].width = 16
ag.freeze_panes = "B2"
wb.active = 0
os.makedirs(os.path.join(OUT, "Models"), exist_ok=True)
wb.save(os.path.join(OUT, "dataset_v3_346_geometries.xlsx"))

# ---- Slide2 models
for gid in ids:
    xr = {f"bnd{b} Angle (°)": h.loc[gid, f"bnd{b}_angle_deg"] for b in range(1, 6)}
    mats = {}
    for k, L in t.SOIL.items():
        v = vals[(gid, L)]   # [γsat, γunsat, cpk, cres, φpk, φres]
        mats[k] = dict(water=HU[L], c=v[3], phi=v[5], c2=v[2], phi2=v[4],
                       ang=round(t.layer_dip(xr, L), 4), uw=v[1], uwbwt=v[0])
    t.write_slim(os.path.join(POOL, pool[gid]), os.path.join(OUT, "Models", f"Feed{gid}.slim"), mats, f"Feed{gid}")
print("built", len(ids))

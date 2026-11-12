"""Read / rewrite the six soil materials inside Slide2 .slim files.
soil1..soil6 = Loam, Blue Clay, Gyttja, Lignite, Black Clay, Green Clay (checked vs Excel)."""
import re, zipfile, io, os
import pandas as pd

SOIL = {1: "Loam", 2: "Blue Clay", 3: "Gyttja", 4: "Lignite", 5: "Black Clay", 6: "Green Clay"}
# layer -> boundaries (above, below); dip = mean of those present
BND = {"Loam": [1], "Blue Clay": [1, 2], "Gyttja": [2, 3], "Lignite": [3, 4],
       "Black Clay": [4, 5], "Green Clay": [5]}

def read_slim(path):
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
        infos = {i.filename: i for i in z.infolist()}
    return files, infos

def sli_name(files):
    return [n for n in files if n.lower().endswith(".sli")][0]

def soil_fields(text, k):
    m = re.search(rf"^  soil{k} = (.*)$", text, re.M)
    toks = m.group(1).split()
    return dict(zip((t.rstrip(":") for t in toks[0::2]), toks[1::2])), m

def set_field(line, key, val):
    new, n = re.subn(rf"(?<=\s){re.escape(key)}: \S+", f"{key}: {val}", " " + line)
    assert n == 1, (key, n)
    return new[1:]

def fmt(v):
    return f"{v:.4f}".rstrip("0").rstrip(".")

def write_slim(src, dst, mats, base_name):
    """mats: {k: dict(c, phi, c2, phi2, ang, uw, uwbwt, water)}"""
    files, infos = read_slim(src)
    old_base = os.path.splitext(sli_name(files))[0]
    sli = sli_name(files)
    text = files[sli].decode("latin-1")
    for k, vals in mats.items():
        _, m = soil_fields(text, k)
        line = m.group(1)
        assert line.startswith("type: 4 "), line[:20]
        for key, v in vals.items():
            line = set_field(line, key, fmt(v))
        text = text[:m.start(1)] + line + text[m.end(1):]
    files[sli] = text.encode("latin-1")
    with zipfile.ZipFile(dst, "w") as z:
        for n, data in files.items():
            if n.lower().endswith(('.s01', '.timing')):   # drop stale results
                continue
            newn = n.replace(old_base, base_name)
            info = infos[n]
            zi = zipfile.ZipInfo(newn, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            z.writestr(zi, data)

def layer_dip(row, layer):
    vals = [float(row[f"bnd{b} Angle (°)"]) for b in BND[layer]]
    return sum(vals) / len(vals)

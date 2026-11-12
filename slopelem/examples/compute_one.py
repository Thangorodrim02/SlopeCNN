"""Critical circle and FoS of one Slide2 model, plus the slice table of that circle.

    python examples/compute_one.py path/to/Feed7.slim
"""

import sys

from slopelem import SlopeModel, bishop_fos, find_critical_circle


def main(path: str) -> None:
    model = SlopeModel.from_file(path)
    res = find_critical_circle(model)
    c = res.circle
    print(f"{path}: FoS = {res.fos:.3f}  ({res.found_by})")
    print(f"  centre ({c.xc:.2f}, {c.yc:.2f}), R = {c.radius:.2f} m, "
          f"entry x = {c.x_entry:.2f}, exit x = {c.x_exit:.2f}")
    for run in res.all_runs:
        print(f"  run {run.method:<13} seed {run.seed:>2}: FoS = {run.fos:.4f}")

    out = bishop_fos(model, c)
    print(f"  min m_alpha = {out.min_m_alpha:.3f}, converged = {out.converged}")
    print("  first 5 slices: x_left  weight(kN/m)  alpha(deg)  c(kPa)  phi(deg)  u(kPa)  material")
    for s in out.slices[:5]:
        print(f"  {s.x_left:10.2f} {s.weight:12.1f} {s.alpha * 57.29578:10.2f} "
              f"{s.cohesion:7.2f} {s.phi:8.2f} {s.pore_pressure:7.1f} {s.material:5d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/data/Feed7.slim")

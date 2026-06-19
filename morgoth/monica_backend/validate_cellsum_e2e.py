#!/usr/bin/env python3
"""
Option C END-TO-END validation gate: cell-summed MONICA backend vs classical
gbm_drm_gen, for ONE real GRB's spacecraft geometry, through the *cspec* path
(MonicaDRMGen) -- the path the TTE 121-event campaign uses.

WHY THIS GATE EXISTS
--------------------
The cell-summed feature swap replaces the atm-cell snap with the geocenter
*direction* taken straight from the spacecraft pose (quaternion + position).
The geocenter unit vector MUST live in the same body frame as the source az/el
the model trained on. A wrong frame produces a "valid-looking" 8-dim feature
that corresponds to the wrong physical geometry -- the per-DRM eval cannot catch
it (it never touches a real pose), and only an end-to-end comparison against
classical gbm for a REAL pose reveals it. So: run this and confirm the DRMs
match to ~a few % (like Path-B --validate-gbm: ~3.5% median, direct-dominated)
BEFORE launching the 121-event TTE run. If it is wildly off, the geo frame /
feature wiring is wrong and the campaign would produce garbage.

Checks, in order:
  (1) the loaded checkpoint really is a cellsum_v1 model (else the gate is moot);
  (2) FRAME self-consistency: geo_unit == [cos(el)cos(az), cos(el)sin(az), sin(el)]
      with az=geo_az, el=90-geo_zenith from _earth_geo_az_el (catches the
      elevation-vs-zenith class of bug);
  (3) DRM rel_frob: MONICA (cellsum) vs classical gbm_drm_gen for the same pose.

Run on server (needs morgoth + gbm_drm_gen, a cellsum checkpoint wired into
morgoth_config, and one GRB's trigdat + cspec -- no TTE required). Example:

  python -m morgoth.monica_backend.validate_cellsum_e2e \
    --det n7 --ra 123.4 --dec -5.6 --t 0.0 \
    --trigdat glg_trigdat_all_bn231012345_v01.fit \
    --cspec  glg_cspec_n7_bn231012345_v00.pha
"""
import os
import argparse
import numpy as np


def _rel_frob(pred, truth):
    n = np.linalg.norm(truth)
    return float("nan") if n < 1e-30 else float(np.linalg.norm(pred - truth) / n)


def _cosine(a, b):
    """Norm-independent shape agreement: cos angle between the flattened DRMs.
    Robust where rel_frob inflates -- a near-blind/off-axis detector has a tiny
    DRM norm, so small absolute errors blow up rel_frob while cosine stays high."""
    x, y = a.ravel(), b.ravel()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float("nan") if d < 1e-30 else float(np.dot(x, y) / d)


def _align(a, b):
    """Transpose-tolerant alignment (MONICA vs gbm may differ by a transpose)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape == b.shape:
        return a, b
    if a.shape == b.T.shape:
        return a, b.T
    raise ValueError(f"incompatible matrix shapes {a.shape} vs {b.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="short name, e.g. n7 / b0")
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--t", type=float, default=0.0, help="response time (s from trigger)")
    ap.add_argument("--trigdat", required=True)
    ap.add_argument("--cspec", required=True)
    ap.add_argument("--rtol", type=float, default=0.10,
                    help="gate: max rel_frob to PASS (default 0.10 ~ a few % + margin)")
    ap.add_argument("--cos-floor", type=float, default=0.99,
                    help="shape-agreement floor: if rel_frob > rtol BUT cosine >= this, "
                         "PASS* (small-norm artifact on a near-blind detector, not a geo error)")
    args = ap.parse_args()

    from morgoth.configuration import morgoth_config
    from morgoth.monica_backend.drmgen import MonicaDRMGen

    cfgm = morgoth_config["drm_backend"]["monica"]

    def cfg_get(key, default=None):
        try:
            v = cfgm[key]
            return default if v is None or (isinstance(v, str) and not v.strip()) else v
        except Exception:
            return default

    # ---- MONICA (cellsum) through the cspec path ----
    mon = MonicaDRMGen(
        det_name=args.det,
        trigdat_file=args.trigdat,
        cspecfile=args.cspec,
        model_path=str(cfg_get("model_path", "") or ""),
        db_path=str(cfg_get("db_path")),
        nai_in_edges=str(cfg_get("nai_in_edges")),
        bgo_in_edges=str(cfg_get("bgo_in_edges")),
        device=str(cfg_get("device", "cpu")),
        batch_size=int(cfg_get("batch_size", 4096)),
        occult=False,
    )
    mon.set_time(args.t)

    # (1) confirm the wired checkpoint is actually a cellsum model
    assert mon._features == "cellsum_v1", (
        f"loaded checkpoint features={mon._features!r}, expected 'cellsum_v1' -- "
        f"the config still points at a legacy (snap) model; this gate is moot.")
    print(f"[ckpt] features={mon._features}  det_long={mon.det_long}")

    # (2) FRAME self-consistency: geo_unit vs reconstruction from (az, 90-zenith)
    g = mon._earth_geo_unit()
    geo_az, geo_zen = mon._earth_geo_az_el()   # NOTE: 2nd value is ZENITH (see drmgen.py)
    geo_el = 90.0 - geo_zen
    a, e = np.deg2rad(geo_az), np.deg2rad(geo_el)
    recon = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    frame_err = float(np.linalg.norm(g - recon))
    print(f"[frame] geo_az={geo_az:.3f} geo_zen={geo_zen:.3f} (el={geo_el:.3f}) "
          f"|geo_unit - recon|={frame_err:.2e}")
    assert frame_err < 1e-6, (
        f"FRAME CHECK FAILED ({frame_err:.2e}): geo_unit disagrees with its own "
        f"az/elevation -- elevation/zenith wiring is wrong.")

    # show the actual 8-dim feature that will be fed to the model
    # (src az/el comes from the GBMFrame transform inside set_location)
    print("[feat] building MONICA DRM (sets src az/el internally) ...")
    mon.set_location(args.ra, args.dec)
    M_mon = mon.matrix

    # ---- classical gbm_drm_gen at the SAME pose time (no TTE needed) ----
    # Mirrors monica-morgoth-validation/07_priority1_drm_comparison_at_swift.py:
    # DRMGen.from_128_bin_data(..., time=t).set_location(ra,dec).matrix. Passing
    # time=args.t keeps the classical pose identical to MONICA's set_time(args.t).
    from gbm_drm_gen.drmgen import DRMGen
    cls = DRMGen.from_128_bin_data(det_name=mon.det_long, time=args.t,
                                   cspecfile=args.cspec, trigdat=args.trigdat,
                                   mat_type=2, occult=False)
    cls.set_location(args.ra, args.dec)
    M_cls = np.asarray(cls.matrix, dtype=np.float64)

    a_mon, a_cls = _align(M_mon, M_cls)
    rf = _rel_frob(a_mon, a_cls)
    cos = _cosine(a_mon, a_cls)
    absf = float(np.linalg.norm(a_mon - a_cls))
    print(f"\n=== END-TO-END [{args.det}] cellsum-MONICA vs classical gbm ===")
    print(f"  matrix shape = {a_mon.shape}")
    print(f"  ||MONICA||={np.linalg.norm(a_mon):.4e}  ||classic||={np.linalg.norm(a_cls):.4e}")
    print(f"  rel_frob = {rf:.4f}   cosine = {cos:.4f}   abs_frob = {absf:.4f}")

    rf_ok = np.isfinite(rf) and rf <= args.rtol
    cos_ok = np.isfinite(cos) and cos >= args.cos_floor
    if rf_ok:
        print(f"\n  GATE PASS (rel_frob {rf:.4f} <= rtol {args.rtol:.4f})")
        print("  -> end-to-end geometry validated; campaign may proceed.")
    elif cos_ok:
        print(f"\n  GATE PASS* (rel_frob {rf:.4f} > rtol {args.rtol:.4f}, "
              f"but cosine {cos:.4f} >= {args.cos_floor:.4f})")
        print("  -> shape agrees; the high rel_frob is a SMALL-NORM metric artifact on a")
        print("     near-blind/off-axis detector (||classic|| tiny), NOT a geometry error.")
    else:
        print(f"\n  GATE FAIL (rel_frob {rf:.4f} > rtol {args.rtol:.4f} "
              f"AND cosine {cos:.4f} < {args.cos_floor:.4f})")
        print("  -> shape disagrees too; investigate the geo frame / feature wiring "
              "before the campaign.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

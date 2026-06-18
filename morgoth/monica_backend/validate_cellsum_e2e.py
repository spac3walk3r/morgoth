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
morgoth_config, and one GRB's trigdat + cspec + tte). Example:

  python -m morgoth.monica_backend.validate_cellsum_e2e \
    --det n7 --ra 123.4 --dec -5.6 --t 0.0 \
    --trigdat glg_trigdat_all_bn231012345_v01.fit \
    --cspec  glg_cspec_n7_bn231012345_v00.pha \
    --tte    glg_tte_n7_bn231012345_v00.fit.gz
"""
import os
import argparse
import numpy as np


def _rel_frob(pred, truth):
    n = np.linalg.norm(truth)
    return float("nan") if n < 1e-30 else float(np.linalg.norm(pred - truth) / n)


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
    ap.add_argument("--tte", required=True)
    ap.add_argument("--rtol", type=float, default=0.10,
                    help="gate: max rel_frob to PASS (default 0.10 ~ a few % + margin)")
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

    # ---- classical gbm_drm_gen, same pose ----
    # NOTE: confirm this accessor matches your installed gbm_drm_gen version --
    # DRMGenTTE(...).set_location(ra, dec) then `.matrix` is the standard path.
    from gbm_drm_gen.drmgen import DRMGenTTE
    cls = DRMGenTTE(tte_file=args.tte, trigdat=args.trigdat, mat_type=2,
                    cspecfile=args.cspec, occult=True)
    cls.set_location(args.ra, args.dec)
    M_cls = np.asarray(cls.matrix, dtype=np.float64)

    a_mon, a_cls = _align(M_mon, M_cls)
    rf = _rel_frob(a_mon, a_cls)
    print(f"\n=== END-TO-END [{args.det}] cellsum-MONICA vs classical gbm ===")
    print(f"  matrix shape = {a_mon.shape}")
    print(f"  ||MONICA||={np.linalg.norm(a_mon):.4e}  ||classic||={np.linalg.norm(a_cls):.4e}")
    print(f"  rel_frob = {rf:.4f}")
    passed = np.isfinite(rf) and rf <= args.rtol
    print(f"\n  GATE {'PASS' if passed else 'FAIL'} (rel_frob {rf:.4f} "
          f"{'<=' if passed else '>'} rtol {args.rtol:.4f})")
    if not passed:
        print("  -> DO NOT launch the 121-event campaign; the geo frame / feature "
              "wiring is wrong.")
        raise SystemExit(1)
    print("  -> end-to-end geometry validated; campaign may proceed.")


if __name__ == "__main__":
    main()

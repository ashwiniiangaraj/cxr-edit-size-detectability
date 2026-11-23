"""
classical-edit size ladder. Re-runs the three IP generators (blob, copy-move, texture
transplant) once per size level, changing only the edit size; all other settings stay at the config.py
"strong" tier, and edge widths scale with the edit (10 % of its diameter).
For each (level, technique): set the size entries of config.IP_CONFIG in place, point config.IP_SYNTHETIC_ROOT
at <IP_size_ladder>/<level>/, run the generator, and restore both in a `finally` block.
Resumable ('.done' markers). Requires config.IP_STRENGTH == "strong". Settings: size_config.py.
Writes <IP_size_ladder>/<level>/<technique>/{train,val,test}/*.png (+ masks, metadata, errors csv).
"""
import copy
import runpy
import time
from pathlib import Path

import config
import size_config as sc

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("generation").resolve()


def script_path(technique):
    """Absolute path of the generator script (the names in size_config are relative to the project root)."""
    return HERE / Path(sc.IP_SCRIPTS[technique]).name


def run_ip_variant(technique, out_root, updates, label=""):
    """Run ONE IP generator with `updates` applied to config.IP_CONFIG[technique], writing under `out_root`.

    updates   dict of config keys -> values, e.g. {"radius_min": 0.124, "radius_max": 0.124}
    Returns the number of images written per split.  config is always restored afterwards.
    """
    cfg = config.IP_CONFIG
    saved, saved_root = copy.deepcopy(cfg), config.IP_SYNTHETIC_ROOT
    try:
        cfg[technique].update(updates)
        config.IP_SYNTHETIC_ROOT = Path(out_root)
        runpy.run_path(str(script_path(technique)), run_name="__main__")
        return {s: len(list((Path(out_root) / technique / s).glob("*.png"))) for s in ("train", "val", "test")}
    finally:
        cfg.clear()
        cfg.update(saved)
        config.IP_SYNTHETIC_ROOT = saved_root          # always restore the real config


def size_updates(technique, level):
    """The size-only change for one technique and level."""
    if technique == "blob":
        r = sc.IP_BLOB_R[level]
        return {"radius_min": r, "radius_max": r}
    frac = sc.IP_SIDE[level] / config.IP_CONFIG[technique]["roi_width"]
    return {"patch_min": frac, "patch_max": frac}


def generate_ladder(levels=None, techniques=None):
    assert config.IP_STRENGTH == "strong", "keep the strong tier so that only the SIZE changes"
    expected = dict(config.SPLIT_IMAGE_COUNTS)
    for level in (levels or sc.IP_LEVELS):
        root = sc.IP_LADDER_ROOT / level
        for tech in (techniques or sc.IP_SCRIPTS):
            if (tech, level) in sc.IP_SKIP:
                print(f"skip {tech} {level} (not feasible: see size_config.IP_SKIP)")
                continue
            if (root / tech / ".done").exists():
                print(f"{tech} {level}: already done")
                continue
            t0 = time.time()
            n = run_ip_variant(tech, root, size_updates(tech, level))
            assert sum(n.values()) > 0, (f"{tech} {level}: nothing written under {root}; "
                                         "the script is not reading config.IP_SYNTHETIC_ROOT")
            (root / tech).mkdir(parents=True, exist_ok=True)
            if n == expected:
                (root / tech / ".done").write_text("ok")
            elif (tech, level) in sc.IP_ACCEPT_PARTIAL:
                (root / tech / ".done").write_text("partial")
                print(f"NOTE {tech} {level}: partial set accepted ({n}); the separation rule rejects some placements")
            else:
                print(f"WARNING {tech} {level}: expected {expected} got {n}; check the errors csv in {root / tech}")
            print(f"{tech} {level}: {n}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    generate_ladder()

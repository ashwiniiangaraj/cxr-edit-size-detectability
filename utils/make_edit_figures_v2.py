"""
Image creation for the report
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ============================== SETTINGS ==============================
DATA = Path(os.environ.get("CXR_PROJECT_ROOT", "/data/aa8624/thesis_2500")) / "data"
OUT_DIR = Path("figures_edits")
SPLIT = "test"
STEM = None                 # image id (part of the file name); None = first id present in every folder
FIG_LEVEL = "5"             # edit size (% of image) for the gallery / pipeline / VAE figures
LEVELS = ["0.5", "1", "2", "5", "10"]
N_STATS = 100               # images used for the inside/outside difference bars
VAE_VMAX = 12.0             # colour scale of the VAE-trace maps, in 8-bit grey levels (|difference|)
CROP = 112                  # side (pixels, of 256) of the crops around each edit in the ladder figure
ZOOM_CROP = 96              # side (pixels, of 256) of the zoomed crops in the gallery
MASK_COLOUR = "#d7263d"
WHOLE_IMAGE_SERIES = ["Diffusion (full)"]   # whole image regenerated: outline drawn dashed ("region requested", not extent of change)
CROP_LADDER_SKIP = ["Diffusion (full)"]      # rows left out of the compact crop ladder (a crop cannot show a whole-image change)
DPI = 300
GRID = 256

REAL_DIR = DATA / "real"
SYNTH = DATA / "synthetic"
CYCLEGAN_DIR = SYNTH / "cyclegan"
RECON_DIR = SYNTH / "diffusion_reconstruction"
IP_LADDER = SYNTH / "IP_size_ladder"              # <a0p5|a1|a2|a5|a10>/<blob|copy|texture>/<split>/*.png (+ masks/<split>)
DIFF_LADDER = SYNTH / "Diffusion_size_ladder"     # <d0p5|d1|d2|d5|d10>/<full|paste>/<split>/*.png, masks/*_edit_mask.png

IP_TAG = {"0.5": "a0p5", "1": "a1", "2": "a2", "5": "a5", "10": "a10"}
DIFF_TAG = {"0.5": "d0p5", "1": "d1", "2": "d2", "5": "d5", "10": "d10"}

# label -> function(level) giving (image folder, mask folder)
SERIES = {
    "Blob":             lambda L: (IP_LADDER / IP_TAG[L] / "blob" / SPLIT,      IP_LADDER / IP_TAG[L] / "blob" / "masks" / SPLIT),
    "Copy-move":        lambda L: (IP_LADDER / IP_TAG[L] / "copy" / SPLIT,      IP_LADDER / IP_TAG[L] / "copy" / "masks" / SPLIT),
    "Texture":          lambda L: (IP_LADDER / IP_TAG[L] / "texture" / SPLIT,   IP_LADDER / IP_TAG[L] / "texture" / "masks" / SPLIT),
    "EditOnly":         lambda L: (DIFF_LADDER / DIFF_TAG[L] / "paste" / SPLIT, DIFF_LADDER / DIFF_TAG[L] / "masks"),
    "Diffusion (full)": lambda L: (DIFF_LADDER / DIFF_TAG[L] / "full" / SPLIT,  DIFF_LADDER / DIFF_TAG[L] / "masks"),
}
# ======================================================================

IMG_EXTS = (".png", ".jpg", ".jpeg")
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})


# ------------------------------- loading -------------------------------
def find(folder, stem):
    """First image in `folder` (recursive) whose name contains `stem`."""
    if folder is None or not Path(folder).exists():
        return None
    hits = sorted(p for p in Path(folder).rglob("*" + stem + "*") if p.suffix.lower() in IMG_EXTS)
    return hits[0] if hits else None


def load_grey(path):
    im = Image.open(path).convert("L")
    if im.size != (GRID, GRID):
        im = im.resize((GRID, GRID), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_mask(path):
    im = Image.open(path).convert("L")
    if im.size != (GRID, GRID):
        im = im.resize((GRID, GRID), Image.NEAREST)
    m = np.asarray(im, dtype=np.float32)
    return m > 0.5 * (m.max() if m.max() > 0 else 1.0)


def series_get(stem, label, level):
    img_dir, mask_dir = SERIES[label](level)
    ip = find(img_dir, stem)
    if ip is None:
        raise SystemExit(f"no image for '{label}' at {level}% (id {stem}) in {img_dir}")
    mp = find(mask_dir, stem)
    return load_grey(ip), (load_mask(mp) if mp is not None else None)


def other_get(stem, folder, name):
    p = find(folder, stem)
    if p is None:
        raise SystemExit(f"no image for {name} (id {stem}) in {folder}")
    return load_grey(p)


def pick_stem():
    if STEM:
        return STEM
    folders = {lab: fn(FIG_LEVEL)[0] for lab, fn in SERIES.items()}
    folders.update({"CycleGAN": CYCLEGAN_DIR, "Reconstruction": RECON_DIR, "Real": REAL_DIR})
    names = {}
    for lab, d in folders.items():
        if not Path(d).exists():
            raise SystemExit(f"folder not found for '{lab}': {d}")
        names[lab] = sorted(p.stem for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not names[lab]:
            raise SystemExit(f"no images for '{lab}' in {d}")
    base = min(names.values(), key=lambda v: len(v[0]))      # the folder with the bare id as file name
    for s in base:
        if all(any(s in n for n in v) for v in names.values()):
            return s
    print("No id appears in every folder. Example file names:")
    for lab, v in names.items():
        print(f"  {lab}: {v[:3]}")
    raise SystemExit("Set STEM to an id that appears in the file names of all folders above.")


# ------------------------------- helpers -------------------------------
def style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def outline(ax, mask, lw=0.8, label=None):
    if mask is not None and mask.any():
        ls = "dashed" if label in WHOLE_IMAGE_SERIES else "solid"
        ax.contour(mask.astype(float), levels=[0.5], colors=MASK_COLOUR, linewidths=lw, linestyles=ls)


KEY_MASK = ("solid", "Solid outline: binary edit mask $M$ (edit area $a$ is computed from $M$)")
KEY_WHOLE = ("dashed", "Dashed outline: region requested for Diffusion (full); the whole image is regenerated")
KEY_ZOOM = ("zoom", "Dashed white box: region shown enlarged in the lower row")


def key(fig, items, loc="upper center", anchor=(0.5, 0.0), ncol=None, fs=6.0):
    """In-figure key so every figure is self-explanatory (needed for the published version)."""
    handles = []
    for kind, text in items:
        if kind == "solid":
            handles.append(Line2D([0], [0], color=MASK_COLOUR, lw=1.3, ls="-", label=text))
        elif kind == "dashed":
            handles.append(Line2D([0], [0], color=MASK_COLOUR, lw=1.3, ls="--", label=text))
        else:
            handles.append(Patch(facecolor="0.35", edgecolor="white", linestyle="--", linewidth=0.9, label=text))
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=ncol or 1, fontsize=fs,
               frameon=False, handlelength=2.4, columnspacing=1.6, borderaxespad=0.2, labelspacing=0.35)


def centre_of(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.mean()), int(xs.mean())


def crop_box(centre, side, mask=None, pad=8):
    """Square crop of `side` px around `centre`; enlarged if needed so that the whole mask fits (+ pad px)."""
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        need = int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1 + 2 * pad)
        side = int(min(GRID, max(side, need)))
    cy, cx = centre
    y0 = int(np.clip(cy - side // 2, 0, GRID - side))
    x0 = int(np.clip(cx - side // 2, 0, GRID - side))
    return y0, x0, side


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", OUT_DIR / f"{name}.pdf")


def collect(stem):
    """All images of one source radiograph, at FIG_LEVEL."""
    real = other_get(stem, REAL_DIR, "Real")
    d = {"Real": (real, None)}
    for lab in SERIES:
        d[lab] = series_get(stem, lab, FIG_LEVEL)
    d["CycleGAN"] = (other_get(stem, CYCLEGAN_DIR, "CycleGAN"), None)
    d["Reconstruction"] = (other_get(stem, RECON_DIR, "Reconstruction"), None)
    return d


# ------------------------------- figures -------------------------------
def fig_gallery(d):
    order = ["Real", "Blob", "Copy-move", "Texture", "CycleGAN", "EditOnly", "Diffusion (full)", "Reconstruction"]
    ref_centre = centre_of(d["Diffusion (full)"][1])           # where the RadEdit edit sits
    fig, axes = plt.subplots(2, len(order), figsize=(7.2, 2.0), squeeze=False)
    for c, lab in enumerate(order):
        img, mask = d[lab]
        centre = centre_of(mask) if mask is not None else ref_centre
        y0, x0, s = crop_box(centre, ZOOM_CROP)
        a = axes[0, c]
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        outline(a, mask, label=lab)
        a.add_patch(Rectangle((x0, y0), s, s, fill=False, ec="white", lw=0.5, ls="--"))
        a.set_title(lab if lab != "Diffusion (full)" else "Diffusion\n(full)", fontsize=6.5, pad=2)
        style(a)
        b = axes[1, c]
        b.imshow(img[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=1)
        style(b)
    axes[0, 0].set_ylabel("Full image", fontsize=6.5)
    axes[1, 0].set_ylabel("Zoom", fontsize=6.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.03, right=0.995, top=0.88, bottom=0.01)
    key(fig, [KEY_MASK, KEY_WHOLE, KEY_ZOOM], anchor=(0.5, 0.0), ncol=1)
    fig.text(0.5, 1.0, f"One test radiograph; edit area {FIG_LEVEL}% of the image (same source image in every column)",
             ha="center", va="bottom", fontsize=6.5)
    save(fig, "fig_gallery")


def vae_stats(stems):
    """Mean |edited - real| (grey levels) inside / outside the edit mask, per series."""
    acc = {k: ([], []) for k in ("EditOnly", "Diffusion (full)", "Reconstruction")}
    for s in stems:
        try:
            real = other_get(s, REAL_DIR, "Real")
            _, mask = series_get(s, "Diffusion (full)", FIG_LEVEL)
            imgs = {"EditOnly": series_get(s, "EditOnly", FIG_LEVEL)[0],
                    "Diffusion (full)": series_get(s, "Diffusion (full)", FIG_LEVEL)[0],
                    "Reconstruction": other_get(s, RECON_DIR, "Reconstruction")}
        except SystemExit:
            continue
        if mask is None or not mask.any():
            continue
        for k, im in imgs.items():
            diff = np.abs(im - real) * 255.0
            acc[k][0].append(diff[mask].mean())
            acc[k][1].append(diff[~mask].mean())
    return {k: (float(np.mean(v[0])), float(np.mean(v[1])), len(v[0])) for k, v in acc.items() if v[0]}


def fig_vae_trace(d, stats):
    labs = ["EditOnly", "Diffusion (full)", "Reconstruction"]
    real, _ = d["Real"]
    mask = d["Diffusion (full)"][1]
    side, gap, W, H = 1.62, 0.08, 7.3, 2.75
    fig = plt.figure(figsize=(W, H))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")
    im = None
    y0 = 0.92
    for i, lab in enumerate(labs):
        ax = fig.add_axes([(0.05 + i * (side + gap)) / W, y0 / H, side / W, side / H])
        diff = np.abs(d[lab][0] - real) * 255.0
        if lab == "Reconstruction":          # no edit: whole-image difference, no grey fill, no outline
            im = ax.imshow(diff, cmap=cmap, vmin=0, vmax=VAE_VMAX)
        else:
            im = ax.imshow(np.ma.masked_where(mask, diff), cmap=cmap, vmin=0, vmax=VAE_VMAX)
            outline(ax, mask, lw=0.6, label="Diffusion (full)")
        ax.set_title(lab, fontsize=7, pad=2)
        style(ax)
    cax = fig.add_axes([0.05 / W, 0.67 / H, 3.0 / W, 0.07 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("|edited \u2212 real| outside the edit region (8-bit grey levels)", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    ax = fig.add_axes([(0.05 + 3 * (side + gap) + 0.62) / W, y0 / H, 1.75 / W, side / H])
    ks = [k for k in labs if k in stats]
    x = np.arange(len(ks))
    ax.bar(x - 0.19, [np.nan if k == "Reconstruction" else stats[k][0] for k in ks], 0.36, color="#4d4d4d", label="inside mask")
    ax.bar(x + 0.19, [stats[k][1] for k in ks], 0.36, color="#e69f00", label="outside mask")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" (full)", "\n(full)") for k in ks], fontsize=6)
    ax.set_ylabel("mean |difference| (grey levels)", fontsize=6.5, labelpad=2)
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(0, 1.35 * max(stats[k][0] for k in ks))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    n = min(stats[k][2] for k in ks)
    ax.set_title(f"Mean over n = {n} images", fontsize=7, pad=2)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_xlabel("Edit family", fontsize=6.5, labelpad=2)
    key(fig, [("dashed", "Dashed outline: edit region (grey, excluded from the maps) in EditOnly and Diffusion (full). Reconstruction has no edit: whole image shown")],
        loc="lower left", anchor=(0.05 / W, 0.0))
    save(fig, "fig_vae_trace")


def fig_ladder_crops(stem):
    labels = [l for l in SERIES if l not in CROP_LADDER_SKIP]
    enlarged = []
    fig, axes = plt.subplots(len(labels), len(LEVELS), figsize=(5.6, 1.1 * len(labels) + 0.45), squeeze=False)
    for r, lab in enumerate(labels):
        for c, L in enumerate(LEVELS):
            img, mask = series_get(stem, lab, L)
            y0, x0, s = crop_box(centre_of(mask), CROP, mask)
            if s != CROP:
                enlarged.append(f"{lab} {L}%")
            ax = axes[r, c]
            ax.imshow(img[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=1)
            sub = mask[y0:y0 + s, x0:x0 + s]
            outline(ax, sub, lw=0.8, label=lab)
            style(ax)
            if r == 0:
                ax.set_title(f"{L}%", fontsize=9, pad=2)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.09, right=0.995, top=0.92, bottom=0.005)
    fig.text(0.5, 0.995, "Edit size (% of image area)", ha="center", va="top", fontsize=9)
    if enlarged:
        print("enlarged crops (mask did not fit; say so in the caption):", enlarged)
    else:
        print("all ladder panels are", CROP, "x", CROP, "px crops")
    key(fig, [("solid", "Solid outline: binary edit mask $M$")], anchor=(0.5, 0.0), fs=8)
    save(fig, "fig_ladder_crops")


def fig_ladder_full(stem):
    labels = list(SERIES)
    fig, axes = plt.subplots(len(labels), len(LEVELS), figsize=(6.3, 1.25 * len(labels) + 0.25), squeeze=False)
    for r, lab in enumerate(labels):
        for c, L in enumerate(LEVELS):
            img, mask = series_get(stem, lab, L)
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            outline(ax, mask, lw=0.7, label=lab)
            style(ax)
            if r == 0:
                ax.set_title(f"{L}%", fontsize=7.5, pad=2)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.07, right=0.995, top=0.93, bottom=0.005)
    fig.text(0.5, 0.995, "Edit size (% of image area)", ha="center", va="top", fontsize=7)
    key(fig, [KEY_MASK, KEY_WHOLE], anchor=(0.5, 0.0))
    save(fig, "fig_ladder_full")


def fig_pipeline(d):
    """Schematic: real radiograph -> synthetic families, with how each family is built."""
    W, H = 7.2, 3.75
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")

    def thumb(x, y, w, img, mask=None, label=None):
        a = fig.add_axes([x / W, y / H, w / W, w / H])
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        outline(a, mask, lw=0.7, label=label)
        style(a)
        for s in a.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color("0.4")
        return a

    def box(x, y, w, h, text, fc="#f2f2f2", fs=6.5, bold=False):
        bg.add_patch(Rectangle((x, y), w, h, fc=fc, ec="0.35", lw=0.5, zorder=2))
        bg.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        bg.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.25", shrinkA=0, shrinkB=0), zorder=1)

    # source image
    tw = 0.78
    thumb(W / 2 - tw / 2, H - tw - 0.12, tw, d["Real"][0])
    bg.text(W / 2 + tw / 2 + 0.08, H - tw / 2 - 0.12, "Real radiograph $X$\n(256$\\times$256, greyscale)", fontsize=6.5, va="center")

    # columns: 7 thumbnails
    items = [("Blob", "IP"), ("Copy-move", "IP"), ("Texture", "IP"), ("CycleGAN", "GAN"),
             ("Reconstruction", "RE"), ("Diffusion (full)", "RE"), ("EditOnly", "RE")]
    caps = {"Blob": "soft disc\nblended", "Copy-move": "patch copied\nwithin image",
            "Texture": "patch from\nanother patient",
            "CycleGAN": "whole-image\ntranslation",
            "Reconstruction": "VAE only:\nVAE$(X)$",
            "Diffusion (full)": "RadEdit output\n$X_D$",
            "EditOnly": "$(1-A)\\odot X+A\\odot X_D$"}
    tws = 0.82
    gap = (W - 0.3 - 7 * tws) / 6
    x0 = 0.15
    ty = 0.95
    centres = {}
    for i, (lab, grp) in enumerate(items):
        x = x0 + i * (tws + gap)
        img, mask = d[lab]
        thumb(x, ty, tws, img, mask, label=lab)
        bg.text(x + tws / 2, ty - 0.08, lab if lab != "Diffusion (full)" else "Diffusion (full)", fontsize=6.3,
                ha="center", va="top", fontweight="bold")
        bg.text(x + tws / 2, ty - 0.30, caps[lab], fontsize=5.6, ha="center", va="top", color="0.25")
        centres[lab] = x + tws / 2

    # group headers
    gy, gh = 2.2, 0.32
    groups = [("Image-processing (IP) edits", ["Blob", "Copy-move", "Texture"]),
              ("GAN", ["CycleGAN"]),
              ("RadEdit (latent diffusion editing)", ["Reconstruction", "Diffusion (full)", "EditOnly"])]
    top_y = H - tw - 0.12
    for title, members in groups:
        xa = centres[members[0]] - tws / 2
        xb = centres[members[-1]] + tws / 2
        box(xa, gy, xb - xa, gh, title, fc="#e8eef7", fs=6.5, bold=True)
        arrow(W / 2, top_y - 0.04, (xa + xb) / 2, gy + gh + 0.03)
        for m in members:
            arrow(centres[m], gy - 0.01, centres[m], ty + tws + 0.02)

    key(fig, [KEY_MASK, KEY_WHOLE], loc="lower center", anchor=(0.5, 0.0))
    bg.text(W / 2, 0.30, "$A$: soft blending mask (slightly wider than $M$); $X_D$: RadEdit output.", fontsize=5.8,
            ha="center", va="bottom", color="0.25")
    save(fig, "fig_pipeline")


def preview(n=12):
    """Contact sheet of candidate radiographs (Blob / Copy-move / Texture / EditOnly at FIG_LEVEL) to choose STEM."""
    stems = sorted(p.stem for p in Path(SERIES["Blob"](FIG_LEVEL)[0]).rglob("*") if p.suffix.lower() in IMG_EXTS)
    stems = stems[:n]
    labs = ["Blob", "Copy-move", "Texture", "EditOnly"]
    fig, axes = plt.subplots(len(stems), len(labs), figsize=(1.6 * len(labs), 1.6 * len(stems)), squeeze=False)
    for r, s in enumerate(stems):
        for c, lab in enumerate(labs):
            try:
                img, mask = series_get(s, lab, FIG_LEVEL)
            except SystemExit:
                img, mask = np.zeros((GRID, GRID)), None
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            outline(ax, mask, lw=0.5, label=lab)
            style(ax)
            if r == 0:
                ax.set_title(lab, fontsize=7)
            if c == 0:
                ax.set_ylabel(s[-12:], fontsize=5)
            if c == 0:
                print(r, s)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    save(fig, "stem_preview")


def main():
    stem = pick_stem()
    print("image id:", stem)
    d = collect(stem)
    fig_gallery(d)
    fig_pipeline(d)
    # all ids present in every folder, for the statistics
    names = sorted(p.stem for p in Path(SERIES["Diffusion (full)"](FIG_LEVEL)[0]).rglob("*") if p.suffix.lower() in IMG_EXTS)
    stats = vae_stats(names[:N_STATS])
    if not stats:
        raise SystemExit("could not compute difference statistics (no common images)")
    for k, (i, o, n) in stats.items():
        print(f"{k:18s} inside {i:6.2f}  outside {o:6.2f}  (grey levels, n={n})")
    fig_vae_trace(d, stats)
    fig_ladder_crops(stem)
    fig_ladder_full(stem)


if __name__ == "__main__":
    import sys
    if "--preview" in sys.argv:          # python make_edit_figures_v2.py --preview 12
        i = sys.argv.index("--preview")
        preview(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 12)
    else:
        main()
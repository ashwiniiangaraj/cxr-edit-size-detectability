# cross_all42.py - handcrafted logistic regression on all 42 features, 5x5 cross-family grid
import numpy as np, pandas as pd, config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

GROUP_COL = config.SCREENING_GROUP_COLUMN
FEATS = list(config.FEATURE_COLUMNS)
assert len(FEATS) == 42 and "lbp_chi2_to_independent_real" not in FEATS, len(FEATS)
LOG10 = [f for f in config.FEATURES_LOG10 if f in FEATS]
assert len(LOG10) == 18, len(LOG10)
C_GRID = list(config.FORENSIC_LOGISTIC_C_GRID)
MAX_ITER = config.FORENSIC_LOGISTIC_MAX_ITER
N_BOOT, SEED = 300, 42

FAMILIES = {
    "IP":             ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant"],
    "CycleGAN":       ["CycleGAN"],
    "Diffusion":      ["Diffusion"],
    "EditOnly":       ["Diffusion_EditOnly"],
    "Reconstruction": ["Diffusion_Reconstruction"],
}

df = pd.read_csv(config.FORENSIC_FEATURE_CSV)
df[LOG10] = np.log10(df[LOG10].clip(lower=0) + config.FEATURE_LOG10_EPS)  
assert df[FEATS].notna().all().all()


def block(split, family):
    real = df[(df[GROUP_COL] == "Real") & (df["split"] == split)]
    syn = df[(df[GROUP_COL].isin(FAMILIES[family])) & (df["split"] == split)]
    X = pd.concat([real, syn])[FEATS].to_numpy()
    y = np.r_[np.zeros(len(real)), np.ones(len(syn))]
    return X, y


def boot_ci(y, s, rng):
    out, n = [], len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


rng = np.random.default_rng(SEED)
rows = []
for A in FAMILIES:
    Xtr, ytr = block("train", A)
    Xva, yva = block("val", A)
    sc = StandardScaler().fit(Xtr)                    
    best = None
    for C in C_GRID:                                  
        m = LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                               max_iter=MAX_ITER, random_state=SEED).fit(sc.transform(Xtr), ytr)
        v = roc_auc_score(yva, m.decision_function(sc.transform(Xva)))
        if best is None or v > best[0]:
            best = (v, C, m)
    val_auc, C_star, model = best
    for B in FAMILIES:                                
        Xte, yte = block("test", B)
        s = model.decision_function(sc.transform(Xte))
        lo, hi = boot_ci(yte, s, rng)
        rows.append(dict(detector="handcrafted_lr_all42", train_family=A, test_family=B,
                         in_family=(A == B), auroc=roc_auc_score(yte, s),
                         ci_lo=lo, ci_hi=hi, C=C_star, val_auroc=val_auc, n_test=len(yte)))

R = pd.DataFrame(rows)
out = config.DET_RESULTS_DIR / "cross_generator_handcrafted_all42.csv"
out.parent.mkdir(parents=True, exist_ok=True)
R.to_csv(out, index=False)

order = list(FAMILIES)
M = R.pivot(index="train_family", columns="test_family", values="auroc").loc[order, order]
print(M.round(3))
print("\nmean in-family :", round(R[R.in_family].auroc.mean(), 3))
print("mean unseen    :", round(R[~R.in_family].auroc.mean(), 3))
print("saved ->", out)

# ---- optional comparison with the stored top-k handcrafted_lr results ----
try:
    old = pd.read_csv(config.DET_RESULTS_CSV)
    o = (old[old["model"] == "handcrafted_lr"]
         .groupby(["train_family", "test_family"])["auroc"].mean())
    n = R.set_index(["train_family", "test_family"])["auroc"]
    cmp = pd.DataFrame({"top_k": o, "all42": n})
    cmp["diff"] = cmp["all42"] - cmp["top_k"]
    print("\n", cmp.round(3))
    print("mean diff (all42 - top_k):", round(cmp["diff"].mean(), 3))
except Exception as e:
    print("comparison skipped:", repr(e))
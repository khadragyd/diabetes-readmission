"""
Uncertainty-aware explainable prediction of 30-day readmission in patients with diabetes
=========================================================================================

Reproduces every result, table and figure in:
  "Beyond Point Predictions: An Uncertainty-Aware Explainable AI Framework for
   Trustworthy 30-Day Hospital Readmission Prediction in Patients With Diabetes
   Using Conformal Prediction and SHAP"

Data: Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository,
      https://doi.org/10.24432/C5230J). Place diabetic_data.csv in ./data/

Usage:
  python run_analysis.py --data data/diabetic_data.csv --out results
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 1000
CV_FOLDS = 5
ALPHAS = [0.20, 0.10, 0.05, 0.01]
EXCLUDED_DISPOSITIONS = [11, 13, 14, 19, 20, 21]   # expired or hospice (IDS_mapping.csv)
DROP_COLS = ["encounter_id", "patient_nbr", "readmitted",
             "weight",                    # ~97% missing
             "examide", "citoglipton"]    # single value in the cohort

SEARCH = {
    "Logistic Regression": dict(n_iter=None, grid={"C": [0.01, 0.1, 1.0, 10.0]}),
    "Random Forest": dict(n_iter=8, grid={
        "max_depth": [8, 12, 16, None], "min_samples_leaf": [5, 20, 50],
        "max_features": ["sqrt", 0.2]}),
    "XGBoost": dict(n_iter=20, grid={
        "n_estimators": [200, 400, 600], "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9], "min_child_weight": [1, 5, 10],
        "reg_lambda": [1.0, 5.0]}),
    "LightGBM": dict(n_iter=20, grid={
        "n_estimators": [200, 400, 600], "num_leaves": [15, 31, 63],
        "learning_rate": [0.02, 0.05, 0.1], "min_child_samples": [20, 50, 100],
        "subsample": [0.8, 1.0], "colsample_bytree": [0.5, 0.7, 0.9]}),
}


# =============================================================== data ======
def icd9_group(code):
    """ICD-9 grouping of Strack et al. (2014)."""
    if code in ("?", "", None):
        return "Missing"
    c = str(code)
    if c.startswith(("V", "E")):
        return "Other"
    try:
        v = float(c)
    except ValueError:
        return "Other"
    i = int(v)
    if 390 <= v <= 459 or i == 785: return "Circulatory"
    if 460 <= v <= 519 or i == 786: return "Respiratory"
    if 520 <= v <= 579 or i == 787: return "Digestive"
    if i == 250: return "Diabetes"
    if 800 <= v <= 999: return "Injury"
    if 710 <= v <= 739: return "Musculoskeletal"
    if 580 <= v <= 629 or i == 788: return "Genitourinary"
    if 140 <= v <= 239: return "Neoplasms"
    return "Other"


def build_cohort(path):
    raw = pd.read_csv(path, keep_default_na=False)
    flow = {"raw_encounters": len(raw), "unique_patients": raw["patient_nbr"].nunique()}
    d = raw.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    n_first = len(d)
    d = d[~d["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)].copy()
    flow.update(first_encounters=n_first, excluded_death_hospice=n_first - len(d),
                final_cohort=len(d))
    y = (d["readmitted"] == "<30").astype(int).to_numpy()
    flow["readmitted_30d"] = int(y.sum())
    flow["prevalence"] = float(y.mean())
    return d, y, flow


def preprocess(d):
    X = d.drop(columns=DROP_COLS).copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        X[c] = X[c].map(icd9_group)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        X[c] = X[c].astype(str)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat:
        X[c] = X[c].replace("?", "Unknown")      # missing -> explicit category
    num = [c for c in X.columns if c not in cat]
    return X, cat, num


def one_hot(train, others, cat):
    clean = lambda df: df.rename(columns=lambda c: str(c).replace("[", "(")
                                 .replace("]", ")").replace("<", "lt").replace(">", "gt"))
    tr = clean(pd.get_dummies(train, columns=cat, dtype=float))
    outs = [clean(pd.get_dummies(o, columns=cat, dtype=float)).reindex(columns=tr.columns, fill_value=0.0)
            for o in others]
    return tr, outs


# ============================================================ metrics ======
def ece(y, p, bins=10):
    """Expected calibration error, 10 equal-frequency bins."""
    order = np.argsort(p)
    err = 0.0
    for b in np.array_split(order, bins):
        err += abs(p[b].mean() - y[b].mean()) * len(b) / len(p)
    return float(err)


def cal_intercept_slope(y, p):
    import statsmodels.api as sm
    lp = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(inter), float(slope)


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    i, s = cal_intercept_slope(y, p)
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                Brier=brier_score_loss(y, p), ECE=ece(y, p), cal_intercept=i, cal_slope=s,
                mean_predicted=float(p.mean()))


def bootstrap_ci(y, p, rng):
    from sklearn.metrics import roc_auc_score, average_precision_score
    a, b = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        a.append(roc_auc_score(y[i], p[i])); b.append(average_precision_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5]).tolist(), np.percentile(b, [2.5, 97.5]).tolist()


def threshold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    ppv, npv = tp / (tp + fp), tn / (tn + fn)
    return dict(threshold=float(t), sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv,
                F1=2 * ppv * sens / (ppv + sens), balanced_accuracy=(sens + spec) / 2)


# ========================================================== conformal ======
def conformal_thresholds(p_cal, y_cal, alpha):
    """Label-conditional split conformal. Score s(x, y) = 1 - p_hat(y | x)."""
    q = {}
    for lab in (0, 1):
        s = (1 - p_cal[y_cal == 1]) if lab == 1 else p_cal[y_cal == 0]
        n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
        q[lab] = np.inf if k > n else float(np.sort(s)[k - 1])
    return q


def conformal_sets(p, q):
    return p <= q[0], (1 - p) <= q[1]          # label 0 included, label 1 included


# =============================================================== main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    import sklearn, xgboost, lightgbm, shap, matplotlib, statsmodels, scipy
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_curve
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from scipy.stats import spearmanr

    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------- cohort ----
    d, y, flow = build_cohort(args.data)
    X, cat, num = preprocess(d)
    print("Cohort:", flow)

    # Table 4: descriptive characteristics
    def pct(mask, grp):
        m = mask[grp]; return f"{int(m.sum()):,} ({100 * m.mean():.1f})"
    def med(v, grp):
        q = np.percentile(v[grp], [50, 25, 75]); return f"{q[0]:.0f} ({q[1]:.0f}–{q[2]:.0f})"
    groups = {"Overall": np.ones(len(y), bool), "Readmitted <30 days": y == 1, "No early readmission": y == 0}
    age70 = d["age"].isin(["[70-80)", "[80-90)", "[90-100)"]).to_numpy()
    rows = {
        "Female, n (%)": (d["gender"] == "Female").to_numpy(),
        "Caucasian, n (%)": (d["race"] == "Caucasian").to_numpy(),
        "African American, n (%)": (d["race"] == "AfricanAmerican").to_numpy(),
        "Age ≥70 years, n (%)": age70,
        "Previous inpatient admission >0, n (%)": (d["number_inpatient"] > 0).to_numpy(),
        "Previous emergency visit >0, n (%)": (d["number_emergency"] > 0).to_numpy(),
        "Previous outpatient visit >0, n (%)": (d["number_outpatient"] > 0).to_numpy(),
        "Diabetes medication prescribed, n (%)": (d["diabetesMed"] == "Yes").to_numpy(),
        "Any insulin use, n (%)": (d["insulin"] != "No").to_numpy(),
        "HbA1c measured, n (%)": (d["A1Cresult"] != "None").to_numpy(),
        "Diabetes medication changed, n (%)": (d["change"] == "Ch").to_numpy(),
    }
    t4 = {k: {g: pct(v, m) for g, m in groups.items()} for k, v in rows.items()}
    for k, col in [("Length of stay, median (IQR), days", "time_in_hospital"),
                   ("Laboratory procedures, median (IQR)", "num_lab_procedures"),
                   ("Number of medications, median (IQR)", "num_medications"),
                   ("Number of diagnoses, median (IQR)", "number_diagnoses")]:
        t4[k] = {g: med(d[col].to_numpy(), m) for g, m in groups.items()}
    pd.DataFrame(t4).T.to_csv(f"{args.out}/table4_characteristics.csv")

    # -------------------------------------------------------- partition ----
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.40, stratify=y, random_state=SEED)
    i_cal, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    Xtr, (Xcal, Xte) = one_hot(X.iloc[i_tr], [X.iloc[i_cal], X.iloc[i_te]], cat)
    ytr, ycal, yte = y[i_tr], y[i_cal], y[i_te]
    part = {"train": [len(i_tr), int(ytr.sum())], "conformal_calibration": [len(i_cal), int(ycal.sum())],
            "test": [len(i_te), int(yte.sum())], "n_features_encoded": Xtr.shape[1]}
    print("Partitions:", part)

    # ------------------------------------------------ models + tuning ----
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    num_idx = [Xtr.columns.get_loc(c) for c in num]
    lr = Pipeline([("scale", ColumnTransformer([("num", StandardScaler(), num_idx)], remainder="passthrough")),
                   ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))])
    estimators = {
        "Logistic Regression": lr,
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(subsample_freq=1, n_jobs=-1, random_state=SEED, verbose=-1),
    }
    tuned, cv_auc = {}, {}
    for name, est in estimators.items():
        spec = SEARCH[name]
        grid = {("clf__" + k if name == "Logistic Regression" else k): v for k, v in spec["grid"].items()}
        if spec["n_iter"] is None:
            s = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=1)
        else:
            s = RandomizedSearchCV(est, grid, n_iter=spec["n_iter"], scoring="roc_auc", cv=cv,
                                   random_state=SEED, n_jobs=1)
        s.fit(Xtr, ytr)
        tuned[name] = s.best_estimator_
        cv_auc[name] = {"cv_auroc": float(s.best_score_),
                        "best_params": {k.replace("clf__", ""): v for k, v in s.best_params_.items()}}
        print(f"{name}: CV AUROC {s.best_score_:.4f} {s.best_params_}  ({time.time() - t0:.0f}s)")

    # ----------------------------------------- calibration of XGBoost ----
    xgb_params = tuned["XGBoost"].get_params()
    cal_xgb = CalibratedClassifierCV(XGBClassifier(**xgb_params), method="sigmoid",
                                     cv=CV_FOLDS, ensemble=True).fit(Xtr, ytr)

    # ------------------------------------------------- test evaluation ----
    preds = {n: m.predict_proba(Xte)[:, 1] for n, m in tuned.items()}
    preds["Calibrated XGBoost"] = cal_xgb.predict_proba(Xte)[:, 1]
    table5 = []
    for n, p in preds.items():
        m = metrics(yte, p)
        ci_a, ci_p = bootstrap_ci(yte, p, np.random.default_rng(SEED))
        m.update(model=n, AUROC_CI=ci_a, AUPRC_CI=ci_p)
        table5.append(m)
    t5 = pd.DataFrame(table5).set_index("model")
    t5.to_csv(f"{args.out}/table5_performance.csv")
    print(t5.round(4).to_string())

    # R2-8: does calibration change ranking?
    base = preds["XGBoost"]; calp = preds["Calibrated XGBoost"]
    platt = LogisticRegression(C=1e6).fit(tuned["XGBoost"].predict(Xtr, output_margin=True).reshape(-1, 1), ytr)
    same = platt.predict_proba(tuned["XGBoost"].predict(Xte, output_margin=True).reshape(-1, 1))[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score
    diffs = []
    r = np.random.default_rng(SEED)
    for _ in range(N_BOOT):
        b = r.integers(0, len(yte), len(yte))
        diffs.append(roc_auc_score(yte[b], calp[b]) - roc_auc_score(yte[b], base[b]))
    r28 = {"spearman_uncal_vs_cv_calibrated": float(spearmanr(base, calp).correlation),
           "spearman_uncal_vs_same_score_sigmoid": float(spearmanr(base, same).correlation),
           "auroc_same_score_sigmoid": float(roc_auc_score(yte, same)),
           "auprc_same_score_sigmoid": float(average_precision_score(yte, same)),
           "auroc_diff_cal_minus_uncal": float(roc_auc_score(yte, calp) - roc_auc_score(yte, base)),
           "auroc_diff_ci": np.percentile(diffs, [2.5, 97.5]).tolist()}
    print("R2-8:", r28)

    # --------------------------------- threshold (from calibration set) ----
    p_cal = cal_xgb.predict_proba(Xcal)[:, 1]
    fpr, tpr, thr = roc_curve(ycal, p_cal)
    t_star = float(thr[np.argmax(tpr - fpr)])          # Youden's J on the calibration partition
    thr_m = threshold_metrics(yte, calp, t_star)
    print("Threshold:", thr_m)

    # ------------------------------------------------------- conformal ----
    conf, groups = [], None
    for al in ALPHAS:
        q = conformal_thresholds(p_cal, ycal, al)
        s0, s1 = conformal_sets(calp, q)
        cov = np.where(yte == 1, s1, s0); size = s0.astype(int) + s1.astype(int)
        conf.append(dict(nominal=1 - al, empirical=cov.mean(), coverage_no=cov[yte == 0].mean(),
                         coverage_yes=cov[yte == 1].mean(), mean_set_size=size.mean(),
                         singleton=(size == 1).mean(), ambiguous=(size == 2).mean(),
                         empty=(size == 0).mean(), n_singleton=int((size == 1).sum()),
                         n_ambiguous=int((size == 2).sum()), q0=q[0], q1=q[1]))
        if abs(al - 0.10) < 1e-9:
            single_lab = np.where(s1 & ~s0, 1, 0)
            groups = np.where(size == 2, "Ambiguous", np.where(size == 0, "Empty",
                     np.where(single_lab == yte, "Confident-correct", "Confident-incorrect")))
    t6 = pd.DataFrame(conf); t6.to_csv(f"{args.out}/table6_conformal.csv", index=False)
    print(t6.round(4).to_string())

    # ------------------------------------------------------------ SHAP ----
    xgb = tuned["XGBoost"]
    sv = shap.TreeExplainer(xgb)(Xte)
    orig = {c: next((k for k in cat if c.startswith(k + "_")), c) for c in Xte.columns}
    agg = pd.DataFrame(sv.values, columns=Xte.columns).T.groupby(orig).sum().T
    imp = agg.abs().mean().sort_values(ascending=False)
    imp.to_csv(f"{args.out}/shap_global_importance.csv", header=["mean_abs_shap"])
    g = pd.Series(groups)
    gsum = pd.DataFrame({"n": g.value_counts(),
                         "mean_predicted_risk": pd.Series(calp).groupby(groups).mean(),
                         "observed_readmission": pd.Series(yte).groupby(groups).mean(),
                         "total_abs_shap": agg.abs().sum(axis=1).groupby(groups).mean(),
                         "share_singleton_readmitted": pd.Series(np.where(groups == "Ambiguous", np.nan, single_lab)).groupby(groups).mean()})
    gsum.to_csv(f"{args.out}/shap_group_summary.csv")
    by_group = agg.abs().groupby(groups).mean()[imp.head(10).index].T
    by_group.to_csv(f"{args.out}/shap_by_group_top10.csv")
    print(gsum.round(3)); print(by_group.round(3))

    # ---------------------------------------------------------- figures ----
    names = list(preds.keys())
    fig, ax = plt.subplots(figsize=(9, 5.2)); w = 0.38; xs = np.arange(len(names))
    b1 = ax.bar(xs - w / 2, t5.loc[names, "AUROC"], w, label="AUROC", color="#1f77b4")
    b2 = ax.bar(xs + w / 2, t5.loc[names, "AUPRC"], w, label="AUPRC", color="#ff7f0e")
    for bars in (b1, b2):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.008, f"{bb.get_height():.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(xs, [n.replace(" ", "\n") for n in names]); ax.set_ylim(0, 0.75)
    ax.set_ylabel("Performance"); ax.set_title("Discrimination across evaluated models"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{args.out}/fig3_discrimination.png", dpi=300); plt.close()

    fig, ax = plt.subplots(figsize=(8.5, 5.5)); nom = t6["nominal"] * 100
    ax.plot(nom, t6["empirical"] * 100, "o-", label="Empirical coverage")
    ax.plot(nom, t6["singleton"] * 100, "o-", label="Singleton predictions")
    ax.plot(nom, t6["ambiguous"] * 100, "o-", label="Ambiguous predictions")
    ax.plot([80, 100], [80, 100], "--", color="grey", label="Nominal = empirical")
    ax.set_xlabel("Nominal coverage (%)"); ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 102)
    ax.set_title("Conformal coverage–informativeness trade-off"); ax.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig4_conformal_tradeoff.png", dpi=300); plt.close()

    plt.figure(); shap.summary_plot(sv.values, Xte, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(f"{args.out}/fig5_shap_beeswarm.png", dpi=300); plt.close()

    agg_expl = shap.Explanation(values=agg.values, base_values=sv.base_values,
                                data=X.iloc[i_te].reindex(columns=agg.columns).values,
                                feature_names=list(agg.columns))
    reps, panels = {}, []
    for grp in ["Confident-correct", "Confident-incorrect", "Ambiguous"]:
        ids = np.where(groups == grp)[0]
        j = ids[np.argmin(abs(calp[ids] - np.median(calp[ids])))]
        reps[grp] = {"test_index": int(j), "calibrated_risk": float(calp[j]), "observed": int(yte[j]),
                     "top_contributions": agg.iloc[j].reindex(agg.iloc[j].abs().sort_values(ascending=False).index[:5]).round(3).to_dict(),
                     "values": X.iloc[i_te[j]][list(agg.iloc[j].abs().sort_values(ascending=False).index[:5])].astype(str).to_dict()}
        plt.figure(); shap.plots.waterfall(agg_expl[j], max_display=8, show=False)
        plt.gcf().set_size_inches(9, 4.2)
        plt.title(f"{grp}: calibrated risk {calp[j]:.3f}, observed outcome "
                  f"{'readmitted' if yte[j] else 'not readmitted'}", fontsize=11)
        fn = f"{args.out}/fig6_waterfall_{grp}.png"
        plt.tight_layout(); plt.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(); panels.append(fn)
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(f).convert("RGB") for f in panels]
    out = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
    yy = 0; dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 70)
    except OSError:
        font = None
    for k, im in enumerate(ims):
        out.paste(im, (0, yy)); dr.text((20, yy + 10), f"({'abc'[k]})", fill="black", font=font); yy += im.height
    out.save(f"{args.out}/fig6_shap_waterfalls.png")

    cols = ["Confident-correct", "Confident-incorrect", "Ambiguous"]
    axg = by_group[cols].iloc[::-1].plot.barh(figsize=(7, 5.5), width=0.8, color=["#2a9d8f", "#e76f51", "#8d99ae"])
    axg.set_xlabel("Mean |SHAP value| (log-odds)"); axg.legend(title="90% conformal group")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig7_shap_by_group.png", dpi=300); plt.close()

    # --------------------------------------------------------- metadata ----
    meta = dict(
        software={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                  "lightgbm": lightgbm.__version__, "shap": shap.__version__,
                  "statsmodels": statsmodels.__version__, "scipy": scipy.__version__},
        seed=SEED, cv_folds=CV_FOLDS, n_bootstrap=N_BOOT, cohort=flow, partitions=part,
        tuning=cv_auc, threshold=thr_m, r2_8_calibration_check=r28,
        representative_patients=reps, runtime_seconds=round(time.time() - t0))
    json.dump(meta, open(f"{args.out}/run_metadata.json", "w"), indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()

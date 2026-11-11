"""
Model-setting experiments for the rainfall notebook.

Runs the notebook's own data loading and feature code, then tries many model settings
on the two validation folds (train before 2022 -> predict 2022, train before 2023 -> predict 2023).
Every result is saved to experiments/experiments.csv. The best settings are written into the
notebook's MODEL SETTINGS cell, and the notebook's experiment log is filled in.

Usage (from the project folder):
    python experiments/tune.py            full search
    python experiments/tune.py --quick    tiny search, only to check that everything runs
    python experiments/tune.py --round2   second search, extending the ranges where the first search's best values sat at the edge
    python experiments/tune.py --readme   refresh the results table in README.md
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

NB = "rainfall_prediction.ipynb"
README = "README.md"
OUT = os.path.join("experiments", "experiments.csv")


def update_readme():
    res = pd.read_csv("validation_results.csv")
    lines = ["## Results", "",
             "Validation RMSE in mm (lower is better). Each year is predicted by models trained only on earlier years.", "",
             "| Model | 2022 | 2023 | Both years |", "|---|---|---|---|"]
    for _, r in res.iterrows():
        name = str(r["model"])
        bold = name == "final"
        label = "**final (submitted blend)**" if bold else name
        vals = [f"{r[c]:.3f}" for c in ["2022", "2023", "both"]]
        if bold:
            vals = [f"**{v}**" for v in vals]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    if os.path.exists(OUT):
        n = len(pd.read_csv(OUT))
        lines += ["", f"Model settings were chosen from {n} configurations tried on both validation years "
                      "(`experiments/tune.py`, results in `experiments/experiments.csv`)."]
    lines.append("")
    text = open(README, encoding="utf-8").read()
    start = text.index("## Results")
    end = text.index("\n## ", start + 5) + 1
    text = text[:start] + "\n".join(lines) + "\n" + text[end:]
    open(README, "w", encoding="utf-8").write(text)
    print("README results table updated")


def main(quick, round2=False):
    t_start = time.time()
    nb = json.load(open(NB, encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    sources = ["".join(c["source"]) for c in code_cells]
    stop = next(i for i, s in enumerate(sources) if s.lstrip().startswith("oof = []"))

    # run the notebook's setup and feature cells
    g = {}
    for s in sources[:stop]:
        exec(compile(s.replace("plt.show()", "plt.close('all')"), NB, "exec"), g)

    T = g["TARGET"]
    HAS_LGB, HAS_CB = g["HAS_LGB"], g["HAS_CB"]
    lgb = g.get("lgb")
    all_feats = g["BASE_FEATS"] + (g["FUTURE_COLS"] if g["USE_FUTURE_ROW"] else [])

    folds = []
    tr_all = g["tr_all"]
    for year in g["VAL_YEARS"]:
        start = pd.Timestamp(f"{year}-01-01")
        tr = tr_all[tr_all.date < start]
        va = tr_all[(tr_all.date >= start) & (tr_all.date < pd.Timestamp(f"{year + 1}-01-01"))]
        va = va[(va.date - start).dt.days % 2 == 0]
        tr, va = g["add_climatology"](tr, [va])
        folds.append((year, tr, va))

    def fit(kind, settings, Xtr, ytr, Xva, yva):
        if kind in ("l2", "tweedie") and HAS_LGB:
            s = dict(settings, objective="tweedie") if kind == "tweedie" else settings
            m = lgb.LGBMRegressor(**g["lgb_params"](42, s))
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
            return m.predict(Xva), m.best_iteration_
        if kind in ("l2", "tweedie"):
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(
                loss="poisson" if kind == "tweedie" else "squared_error", max_iter=150, learning_rate=0.1,
                max_leaf_nodes=settings["num_leaves"], min_samples_leaf=settings["min_child_samples"], random_state=42)
            m.fit(Xtr, ytr)
            return m.predict(Xva), m.n_iter_
        if kind == "cat":
            m = g["CatBoostRegressor"](iterations=3000, learning_rate=0.08, loss_function="RMSE", random_seed=42,
                                       verbose=False, thread_count=-1, **settings)
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=200)
            return m.predict(Xva), m.get_best_iteration()
        raise ValueError(kind)

    # keep earlier rounds' results in the same file
    rows = []
    if round2 and os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        prev["round"] = prev["round"].fillna(1).astype(int) if "round" in prev else 1
        rows = prev.to_dict("records")

    def evaluate(stage, kind, settings, drop=()):
        feats = [f for f in all_feats if f not in drop]
        t0 = time.time()
        sse, n, per, iters = 0.0, 0, {}, []
        for year, tr, va in folds:
            p, it = fit(kind, settings, tr[feats], tr[T], va[feats], va[T])
            err = (np.clip(p, 0, None) - va[T].values) ** 2
            per[year] = float(np.sqrt(err.mean()))
            sse += err.sum()
            n += len(err)
            iters.append(it)
        both = float(np.sqrt(sse / n))
        row = {"round": 2 if round2 else 1, "stage": stage, "model": kind, "settings": json.dumps(settings, sort_keys=True),
               "dropped_features": ",".join(drop),
               **{f"rmse_{y}": round(v, 4) for y, v in per.items()},
               "rmse_both": round(both, 4), "iterations": "/".join(str(i) for i in iters),
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        os.makedirs("experiments", exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[{stage}] {kind:8s} {row['settings']} drop=[{row['dropped_features']}] "
              f"-> both {both:.4f}  " + "  ".join(f"{y}: {v:.4f}" for y, v in per.items()), flush=True)
        return both, per

    # ---- 1. LightGBM (squared error): tree size, leaf size, column sampling ----
    current_l2 = dict(g["LGB_L2"])
    current_drop = list(g["DROP_FEATS"])
    base = dict(learning_rate=0.03, reg_lambda=1.0)
    if quick:
        grid = [(127, 50, 0.7), (31, 200, 0.7)]
    elif round2:
        grid = [(nl, mc, cs) for nl in (63, 127, 255) for mc in (500, 1000, 2000) for cs in (0.3, 0.5)]
        base = dict(learning_rate=0.03, reg_lambda=current_l2.get("reg_lambda", 20.0))
    else:
        grid = [(nl, mc, cs) for nl in (15, 31, 63, 127) for mc in (50, 200, 500) for cs in (0.5, 0.8)]
    cur_score, cur_per = evaluate("current", "l2", current_l2, current_drop)
    best_l2, best_score = current_l2, cur_score
    for nl, mc, cs in grid:
        s = dict(base, num_leaves=nl, min_child_samples=mc, colsample_bytree=cs)
        if s == current_l2:
            continue
        score, _ = evaluate("l2-grid", "l2", s, current_drop)
        if score < best_score:
            best_l2, best_score = s, score

    # ---- 2. regularisation on the best tree settings ----
    for lam in ([] if quick else ((20.0, 50.0, 100.0, 200.0) if round2 else (0.0, 5.0, 20.0))):
        s = dict(best_l2, reg_lambda=lam)
        score, _ = evaluate("l2-reg", "l2", s, current_drop)
        if score < best_score:
            best_l2, best_score = s, score

    # ---- 3. dropping feature groups that might not generalise across years ----
    drop_sets = [["station_name_cat"]] if quick else [
        ["station_name_cat"], ["station_name_cat", "state_cat"], ["nb_count", "nb_min_km"],
        ["india_rain_mean", "india_wet_share"], ["rain_lag6", "temp_chg2"]]
    best_drop = current_drop
    for d in drop_sets:
        drop = sorted(set(current_drop) | set(d))
        score, _ = evaluate("features", "l2", best_l2, drop)
        if score < best_score - 0.002:          # only drop features for a clear gain
            best_drop, best_score = drop, score

    # ---- 4. Tweedie power, using the best tree settings ----
    best_tw, best_tw_score = None, np.inf
    for power in ((1.3,) if quick else ((1.02, 1.05, 1.1, 1.2) if round2 else (1.1, 1.2, 1.3, 1.4, 1.5))):
        s = dict(best_l2, tweedie_variance_power=power)
        score, _ = evaluate("tweedie", "tweedie", s, best_drop)
        if score < best_tw_score:
            best_tw, best_tw_score = s, score

    # ---- 5. CatBoost depth and L2 regularisation ----
    best_cat, best_cat_score = dict(g["CAT_PARAMS"]), np.inf
    if HAS_CB and round2 and not quick:
        best_cat_score, _ = evaluate("current", "cat", best_cat, best_drop)
    if HAS_CB and not quick:
        for depth in ((10, 12) if round2 else (6, 8, 10)):
            for l2r in ((1.0, 3.0, 10.0) if round2 else (3.0, 10.0)):
                s = dict(depth=depth, l2_leaf_reg=l2r)
                score, _ = evaluate("catboost", "cat", s, best_drop)
                if score < best_cat_score:
                    best_cat, best_cat_score = s, score

    # ---- write the chosen settings into the notebook ----
    fmt = lambda d: "dict(" + ", ".join(f"{k}={v!r}" for k, v in d.items()) + ")"
    settings_cell = "\n".join([
        "# MODEL SETTINGS",
        "# chosen by experiments/tune.py on the 2022 + 2023 validation folds (see experiments/experiments.csv)",
        f"LGB_L2 = {fmt(best_l2)}",
        f"LGB_TWEEDIE = {fmt(best_tw)}",
        f"CAT_PARAMS = {fmt(best_cat)}",
        f"DROP_FEATS = {best_drop!r}",
        "FEATS = [f for f in FEATS if f not in DROP_FEATS]",
        'print(len(FEATS), "features used")',
    ])
    df = pd.DataFrame(rows)
    log = ["## 10. Experiment log", "",
           f"`experiments/tune.py` tried **{len(df)} configurations** on both validation years "
           "(full list in `experiments/experiments.csv`). RMSE of single models, before blending:", "",
           "| model | settings | 2022 | 2023 | both |", "|---|---|---|---|---|"]
    df["dropped_features"] = df["dropped_features"].fillna("")
    picks = [("LightGBM before tuning", df.iloc[0])]
    for label, kind in [("LightGBM after tuning", "l2"), ("Tweedie (best power)", "tweedie"), ("CatBoost (best)", "cat")]:
        sub = df[(df.model == kind) & (df.dropped_features.fillna("") == ",".join(best_drop))]
        if len(sub):
            picks.append((label, sub.loc[sub.rmse_both.idxmin()]))
    for label, r in picks:
        yrs = [c for c in df.columns if c.startswith("rmse_") and c != "rmse_both"]
        drop = f", dropped: {r['dropped_features']}" if r["dropped_features"] else ""
        log.append(f"| {label} | `{r['settings']}`{drop} | " + " | ".join(f"{r[c]:.3f}" for c in yrs)
                   + f" | {r['rmse_both']:.3f} |")
    log += ["", "Rule used: a change is kept only if it lowers the RMSE pooled over both validation years "
                "(features are dropped only for a gain above 0.002)."]

    found_settings = found_log = False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.startswith("# MODEL SETTINGS"):
            c["source"] = settings_cell.splitlines(keepends=True)
            c["outputs"], c["execution_count"] = [], None
            found_settings = True
        if c["cell_type"] == "markdown" and src.startswith("## 10. Experiment log"):
            c["source"] = "\n".join(log).splitlines(keepends=True)
            found_log = True
    assert found_settings and found_log, "notebook cells not found"
    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("\nchosen settings:\n" + settings_cell)
    print(f"\nLightGBM RMSE (both years): {cur_score:.4f} -> {best_score:.4f}")
    print(f"total time {(time.time() - t_start) / 60:.1f} min, {len(df)} configurations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--readme", action="store_true")
    ap.add_argument("--round2", action="store_true")
    a = ap.parse_args()
    if a.readme:
        update_readme()
    else:
        main(a.quick, a.round2)

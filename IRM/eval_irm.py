# eval_irm.py
# coding: utf-8
import os, json, argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from irm_iclr import (
    DataConfig, make_examples, stratified_split_by_score_bins,
    IRMScorer
)

def load_year_z_stats(model_dir):
    path = os.path.join(model_dir, "year_z_stats.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    stats = obj.get("year_stats", {})
    default = obj.get("default", {"mean": 0.0, "std": 1.0})
    stats = {
        (None if k in ["", "-1", "None", None] else int(k)):
            (float(v["mean"]), float(v["std"])) for k, v in stats.items()
    }
    mu_all = float(default.get("mean", 0.0))
    sd_all = float(default.get("std", 1.0))
    if sd_all <= 1e-8: sd_all = 1.0
    return stats, (mu_all, sd_all)

def apply_year_z_with_stats(scores, years, stats, default_stats):
    out = []
    for s, y in zip(scores, years):
        y2 = None if (y in ["", None]) else int(y)
        mu, sd = stats.get(y2, default_stats)
        if sd <= 1e-8: sd = 1e-8
        out.append( (float(s) - float(mu)) / float(sd) )
    return np.array(out, dtype=float)

def best_f1_operating_point(y_true, y_score):
    pr_ths = np.linspace(y_score.min(), y_score.max(), num=200)
    best = (0.0, 0.0, 0.0, 0.0)  # th, P, R, F1
    for th in pr_ths:
        y_pred = (y_score >= th).astype(int)
        if y_pred.sum() == 0 and y_true.sum() == 0:
            continue
        P = ( (y_pred & (y_true==1)).sum() / max(y_pred.sum(), 1) )
        R = ( (y_pred & (y_true==1)).sum() / max((y_true==1).sum(), 1) )
        F1 = 0.0 if (P+R)==0 else (2*P*R/(P+R))
        if F1 > best[3]:
            best = (th, P, R, F1)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--target_type", default="year_z", choices=["raw","year_z"])
    ap.add_argument("--accept_threshold", type=float, default=6.0)
    ap.add_argument("--add_year_tag", action="store_true")
    ap.add_argument("--eval_subset", default="all", choices=["all","valid","train"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--use_sliding", action="store_true")
    ap.add_argument("--use_reward", action="store_true", help="0-1のrewardで2値評価も出す")
    ap.add_argument("--stride_ratio", type=float, default=0.5)
    ap.add_argument("--agg", default="median", choices=["mean","median","wmean"])
    args = ap.parse_args()

    # データ用の examples を作る（train/validの分割再現のため）
    dcfg = DataConfig(
        data_path=args.data_path,
        max_length=args.max_len,
        target_type="raw",  # まず raw を作っておき、year_zは学習時統計で後から変換
        accept_threshold=args.accept_threshold,
        add_year_tag=args.add_year_tag
    )
    examples = make_examples(dcfg)
    train_ex, valid_ex = stratified_split_by_score_bins(examples, train_ratio=0.9, seed=dcfg.seed)
    if args.eval_subset == "valid":
        ex = valid_ex
    elif args.eval_subset == "train":
        ex = train_ex
    else:
        ex = examples

    texts = [e["text"] for e in ex]
    raw_scores = np.array([e["score"] for e in ex], dtype=float)
    years = np.array([e["year"] if e["year"] is not None else None for e in ex], dtype=object)
    accepts = np.array([1 if s >= args.accept_threshold else 0 for s in raw_scores], dtype=int)

    # ラベル作成（year_z が要求された場合は、学習時の μ/σ で変換）
    if args.target_type == "year_z":
        zstats = load_year_z_stats(args.model_dir)
        if zstats is None:
            raise RuntimeError("year_z で評価するには model_dir に year_z_stats.json が必要です。学習側で保存してください。")
        stats, default_stats = zstats
        targets = apply_year_z_with_stats(raw_scores, years, stats, default_stats)
    else:
        targets = raw_scores.copy()

    # 推論
    scorer = IRMScorer(args.model_dir, max_length=args.max_len,
                       window_stride_ratio=args.stride_ratio, agg=args.agg)
    if args.use_sliding:
        preds = np.array([scorer._score_single(t) for t in tqdm(texts)], dtype=float)
    else:
        # 高速バッチ推論（非スライディング）
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device); mdl.eval()
        preds_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), args.bs)):
                batch_texts = texts[i:i+args.bs]
                batch = tok(batch_texts, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = mdl(**batch)
                ps = out.logits.squeeze(-1).detach().cpu().numpy().tolist()
                preds_list.extend(ps)
        preds = np.array(preds_list, dtype=float)

    # 回帰指標
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))
    try:
        from scipy.stats import spearmanr
        spr  = float(spearmanr(preds, targets).correlation)
    except Exception:
        spr = 0.0

    print("\n[Regression metrics]")
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  Spearman={spr:.4f}")

    # 2値（accept）指標（回帰出力をスコアとみなして）
    try:
        roc_auc = float(roc_auc_score(accepts, preds))
        pr_auc  = float(average_precision_score(accepts, preds))
    except Exception:
        roc_auc = pr_auc = 0.0
    print("\n[Binary (Accept) metrics from regression score]")
    print(f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    th, P, R, F1 = best_f1_operating_point(accepts, preds)
    y_pred = (preds >= th).astype(int)
    TP = int(((y_pred==1) & (accepts==1)).sum())
    FP = int(((y_pred==1) & (accepts==0)).sum())
    TN = int(((y_pred==0) & (accepts==0)).sum())
    FN = int(((y_pred==0) & (accepts==1)).sum())
    print("\n[Operating point @ best F1]")
    print(f"th={th:.3f}  P={P:.3f}  R={R:.3f}  F1={F1:.3f}  TP={TP} FP={FP} TN={TN} FN={FN}")

    # キャリブレーション（10ビン）
    qs = np.quantile(preds, np.linspace(0, 1, 11))
    print("\n[Calibration buckets] accept | score=regression_output")
    print("bin\tpred_lo\tpred_hi\tn\ttrue_mean\tpred_mean")
    for i in range(10):
        lo, hi = qs[i], qs[i+1]
        idx = (preds >= lo) & (preds <= hi if i==9 else preds < hi)
        n = int(idx.sum())
        tmean = float(accepts[idx].mean()) if n>0 else 0.0
        pmean = float(preds[idx].mean()) if n>0 else 0.0
        print(f"{i}\t{lo:.3f}\t{hi:.3f}\t{n}\t{tmean:.3f}\t\t{pmean:.3f}")

    # 年別 RMSE/ρ
    years_arr = np.array([ (y if y is not None else -1) for y in years ])
    uniq_years = [y for y in sorted(set(years_arr)) if y != -1]
    print("\n[Per-year metrics]")
    print("year\tn\trmse\tspearman")
    for y in uniq_years:
        idx = (years_arr == y)
        if idx.sum() == 0: continue
        rmse_y = float(np.sqrt(np.mean((preds[idx] - targets[idx]) ** 2)))
        try:
            from scipy.stats import spearmanr
            spr_y = float(spearmanr(preds[idx], targets[idx]).correlation)
        except Exception:
            spr_y = 0.0
        print(f"{int(y)}\t{int(idx.sum())}\t{rmse_y:.3f}\t{spr_y:.3f}")

    # reward(0-1) での2値評価（オプション）
    if args.use_reward:
        # reward は percentile-calibration 由来なので単調変換→AUC類は同じになるが、PRは微妙に変わることあり
        scorer2 = IRMScorer(args.model_dir, max_length=args.max_len,
                            window_stride_ratio=args.stride_ratio, agg=args.agg)
        reward = np.array([scorer2._to_reward(x) for x in preds], dtype=float)
        try:
            roc_auc2 = float(roc_auc_score(accepts, reward))
            pr_auc2  = float(average_precision_score(accepts, reward))
        except Exception:
            roc_auc2 = pr_auc2 = 0.0
        # spearman(reward, accept)
        try:
            from scipy.stats import spearmanr
            spr_acc = float(spearmanr(reward, accepts).correlation)
        except Exception:
            spr_acc = 0.0

        print("\n[Binary (Accept) metrics from reward (0-1, percentile-calibrated)]")
        print(f"ROC-AUC={roc_auc2:.4f}  PR-AUC={pr_auc2:.4f}  Spearman(reward, accept)={spr_acc:.4f}")

        qs2 = np.quantile(reward, np.linspace(0,1,11))
        print("\n[Calibration buckets] accept | score=reward(0-1)")
        print("bin\tpred_lo\tpred_hi\tn\ttrue_mean\tpred_mean")
        for i in range(10):
            lo, hi = qs2[i], qs2[i+1]
            idx = (reward >= lo) & (reward <= hi if i==9 else reward < hi)
            n = int(idx.sum())
            tmean = float(accepts[idx].mean()) if n>0 else 0.0
            pmean = float(reward[idx].mean()) if n>0 else 0.0
            print(f"{i}\t{lo:.3f}\t{hi:.3f}\t{n}\t{tmean:.3f}\t\t{pmean:.3f}")

    # 保存
    out = {
        "rmse": rmse, "mae": mae, "spearman": spr,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }
    out_path = os.path.join(args.model_dir, f"eval_summary_{args.eval_subset}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\n[Saved] {out_path}")

if __name__ == "__main__":
    main()


# 使い方
# python eval_irm.py \
#   --model_dir ./irm_sci_huber_z \
#   --data_path data/iclr/ \
#   --target_type year_z --accept_threshold 6.0 --add_year_tag \
#   --eval_subset valid \
#   --bs 64 --max_len 512
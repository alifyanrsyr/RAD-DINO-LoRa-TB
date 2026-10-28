from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = ROOT / "output" / "cross_validation" / "08_e7_oof_model_selection_v1"
LOCK = OOF_DIR / "08_e7_oof_model_selection_LOCKED.json"
TRAIN = ROOT / "output" / "training"
OUT = ROOT / "output" / "overfitting" / "10_e7_overfitting_review_v1"
MODS = ("frozen_raw", "frozen_clahe", "lora_raw", "lora_clahe")
REQUIRED_COLS = (
    "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_auroc"
)
STATUS = "PASS_E7_OVERFITTING_DIAGNOSTIC_REVIEW"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    require(path.is_file(), f"File tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect():
    lock = read_json(LOCK)
    require(lock.get("status") == "PASS_E7_OOF_MODEL_SELECTION_LOCKED",
            "OOF lock E7 tidak valid")
    require(lock.get("n_training_runs") == 20 and lock.get("n_models") == 4,
            "Jumlah model/run bukan protokol E7 (20 run)")
    require(lock.get("primary_model") == "lora_clahe" and
            lock.get("primary_selected_before_e7_test") is True,
            "Pemilihan model utama tidak sesuai OOF lock")
    for key in ("test_pixel_read", "test_predictions_read", "test_inference_performed"):
        require(lock.get(key) is False, f"OOF lock tidak menjamin {key}=false")
    refs = lock.get("run_artifact_sha256", {})
    rows = []
    histories = {}
    summaries = {}
    for mod in MODS:
        for fold in range(1, 6):
            key = f"{mod}_fold_{fold}"
            run_dir = TRAIN / key
            hist_path = run_dir / "training_history.csv"
            summary_path = run_dir / "run_summary.json"
            require(hist_path.is_file() and summary_path.is_file(),
                    f"Belum ditemukan training_history.csv atau run_summary.json: {run_dir}")
            expected = refs.get(key)
            require(isinstance(expected, dict), f"Hash lock untuk {key} tidak ditemukan")
            require(sha256(hist_path) == expected.get("training_history.csv"),
                    f"Hash history tidak sesuai OOF lock: {key}")
            require(sha256(summary_path) == expected.get("run_summary.json"),
                    f"Hash summary tidak sesuai OOF lock: {key}")
            hist = pd.read_csv(hist_path)
            require(set(REQUIRED_COLS).issubset(hist.columns),
                    f"Kolom training history tidak lengkap: {key}")
            require(len(hist) >= 1, f"Training history kosong: {key}")
            for name in REQUIRED_COLS:
                hist[name] = pd.to_numeric(hist[name], errors="raise")
                require(np.isfinite(hist[name].to_numpy(dtype=float)).all(),
                        f"Nilai tidak finite pada {name}: {key}")
            require(np.array_equal(hist.epoch.to_numpy(), np.arange(1, len(hist)+1)),
                    f"Urutan epoch tidak kontinu dimulai dari 1: {key}")
            sm = read_json(summary_path)
            best_epoch = int(sm["best_epoch"])
            last_epoch = int(sm["last_epoch"])
            require(1 <= best_epoch <= last_epoch == len(hist),
                    f"Best epoch/last epoch tidak sesuai history: {key}")
            b = hist.iloc[best_epoch - 1]
            last = hist.iloc[-1]
            best_auc_reported = float(sm["best_validation_auroc"])
            require(math.isclose(float(b.val_auroc), best_auc_reported,
                                 rel_tol=1e-8, abs_tol=1e-8),
                    f"Best validation AUROC tidak sesuai history: {key}")
            for c in ("train_accuracy", "val_accuracy", "val_auroc"):
                require(hist[c].between(0, 1).all(), f"Rentang {c} tidak valid: {key}")
            require((hist.train_loss >= 0).all() and (hist.val_loss >= 0).all(),
                    f"Loss bernilai negatif: {key}")
            train_loss_delta = float(last.train_loss - b.train_loss)
            val_loss_delta = float(last.val_loss - b.val_loss)
            val_auc_delta = float(last.val_auroc - b.val_auroc)
            n_after = last_epoch - best_epoch
            diverging = (n_after > 0 and train_loss_delta < 0 and
                         val_loss_delta > 0 and val_auc_delta < 0)
            # Heuristic directional indicator, not proof of overfitting.
            rows.append({
                "model": mod, "fold": fold,
                "best_epoch": best_epoch, "last_epoch": last_epoch,
                "epochs_after_best": n_after,
                "best_val_auroc": float(b.val_auroc),
                "last_val_auroc": float(last.val_auroc),
                "delta_val_auroc_after_best": val_auc_delta,
                "best_train_loss": float(b.train_loss),
                "last_train_loss": float(last.train_loss),
                "delta_train_loss_after_best": train_loss_delta,
                "best_val_loss": float(b.val_loss),
                "last_val_loss": float(last.val_loss),
                "delta_val_loss_after_best": val_loss_delta,
                "train_accuracy_at_best": float(b.train_accuracy),
                "val_accuracy_at_best": float(b.val_accuracy),
                "accuracy_gap_at_best": float(b.train_accuracy - b.val_accuracy),
                "min_val_loss_epoch": int(hist.loc[hist.val_loss.idxmin(), "epoch"]),
                "trend_divergence_after_best": bool(diverging),
                "interpretation": "indikasi_divergensi" if diverging else "tidak_terdeteksi_dengan_heuristik",
            })
            histories[key] = hist
            summaries[key] = sm
    details = pd.DataFrame(rows)
    require(len(details) == 20 and details[["model", "fold"]].drop_duplicates().shape[0] == 20,
            "Jumlah history tidak sesuai 20 run")
    grouped = details.groupby("model", sort=False).agg(
        folds=("fold", "count"),
        best_epoch_mean=("best_epoch", "mean"),
        last_epoch_mean=("last_epoch", "mean"),
        mean_best_val_auroc=("best_val_auroc", "mean"),
        mean_val_loss_delta_after_best=("delta_val_loss_after_best", "mean"),
        mean_val_auroc_delta_after_best=("delta_val_auroc_after_best", "mean"),
        mean_accuracy_gap_at_best=("accuracy_gap_at_best", "mean"),
        n_trend_divergence_flags=("trend_divergence_after_best", "sum"),
    ).reset_index()
    grouped["n_trend_divergence_flags"] = grouped["n_trend_divergence_flags"].astype(int)
    return lock, details, grouped, histories, summaries


def generate_figures(details: pd.DataFrame, histories: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    for mod in MODS:
        fig, axs = plt.subplots(3, 5, figsize=(17, 9), dpi=170)
        for col in range(5):
            fold = col + 1
            key = f"{mod}_fold_{fold}"
            h = histories[key]
            best = int(details.loc[(details.model == mod) & (details.fold == fold), "best_epoch"].iloc[0])
            ax = axs[0, col]
            ax.plot(h.epoch, h.train_loss, label="Training loss", linewidth=1.8)
            ax.plot(h.epoch, h.val_loss, label="Validation loss", linewidth=1.8)
            ax.axvline(best, linestyle="--", alpha=0.55, color="gray")
            ax.set_title(f"Fold {fold} | Best epoch {best}", fontsize=11)
            ax.grid(alpha=0.2)
            if col == 0: ax.set_ylabel("Loss")
            ax = axs[1, col]
            ax.plot(h.epoch, h.train_accuracy, label="Training accuracy", linewidth=1.8)
            ax.plot(h.epoch, h.val_accuracy, label="Validation accuracy", linewidth=1.8)
            ax.axvline(best, linestyle="--", alpha=0.55, color="gray")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            if col == 0: ax.set_ylabel("Accuracy")
            ax = axs[2, col]
            ax.plot(h.epoch, h.val_auroc, label="Validation AUROC", linewidth=1.8)
            ax.axvline(best, linestyle="--", alpha=0.55, color="gray")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Epoch")
            if col == 0: ax.set_ylabel("Validation AUROC")
        handles, labels = [], []
        for ax in axs.flatten():
            hh, ll = ax.get_legend_handles_labels()
            for h, l in zip(hh, ll):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
        fig.suptitle(f"Eksperimen 7 | Training Dynamics — {mod.replace('_', ' ').upper()}",
                     fontsize=17, fontweight="bold")
        fig.tight_layout(rect=[0.02, 0.075, 0.98, 0.95])
        name = f"10_e7_training_dynamics_{mod}.png"
        fig.savefig(out / name, dpi=170)
        plt.close(fig)
        figures.append(name)
    return figures


def display(details, grouped):
    print("\nE7 | ANALISIS OVERFITTING | DEVELOPMENT ONLY")
    print(f"History valid: {len(details)}/20 | Model utama tidak berubah: lora_clahe")
    cols = ["model", "best_epoch_mean", "last_epoch_mean", "mean_best_val_auroc",
            "mean_val_loss_delta_after_best", "mean_val_auroc_delta_after_best",
            "n_trend_divergence_flags"]
    print(grouped[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n'Indikasi divergensi' = setelah best epoch, train loss turun, val loss naik, val AUROC turun.")
    print("Heuristik deskriptif, BUKAN pembuktian kausal/statistik overfitting.")
    print("Tidak menggunakan dataset testing atau mengubah threshold/model utama.")


def run():
    lock, details, grouped, histories, _ = collect()
    display(details, grouped)
    require(not OUT.exists(), f"Folder output sudah ada; jangan timpa: {OUT}. Gunakan --verify.")
    OUT.mkdir(parents=True, exist_ok=False)
    files = []
    for name, df in (("10_e7_overfitting_per_fold.csv", details),
                     ("10_e7_overfitting_by_model.csv", grouped)):
        df.to_csv(OUT / name, index=False, float_format="%.12g")
        files.append(name)
    files += generate_figures(details, histories, OUT)
    report = {
        "status": STATUS,
        "experiment": "experiment7",
        "model_selection_locked": lock["primary_model"],
        "run_count": 20,
        "n_models": 4,
        "n_folds_per_model": 5,
        "overfitting_flag_is_heuristic": True,
        "rule": "best<last AND train_loss(last)<train_loss(best) AND val_loss(last)>val_loss(best) AND val_auroc(last)<val_auroc(best)",
        "n_divergence_flags": int(details.trend_divergence_after_best.sum()),
        "source_oof_lock_sha256": sha256(LOCK),
        "no_test_data_read": True,
        "no_training": True,
        "no_inference": True,
        "limitations": [
            "Directional pattern after best epoch is a screening indicator, not a statistical test.",
            "Checkpoint was selected on validation AUROC: descriptive optimism possible.",
            "Changing multiple E7 training hyperparameters together does not identify causal parameter effects.",
            "Testing cohort was reused in earlier experiments; future test comparison is exploratory."
        ],
        "output_sha256": {name: sha256(OUT / name) for name in files}
    }
    report_name = "10_e7_overfitting_report.json"
    (OUT / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{STATUS} | Output: {OUT}")


def verify():
    report = read_json(OUT / "10_e7_overfitting_report.json")
    require(report["status"] == STATUS, "Status laporan salah")
    require(report["source_oof_lock_sha256"] == sha256(LOCK), "OOF lock berubah")
    for name, expected in report["output_sha256"].items():
        require((OUT / name).is_file() and sha256(OUT / name) == expected,
                f"Output tidak sesuai hash: {name}")
    _, details, grouped, _, _ = collect()
    require(int(details.trend_divergence_after_best.sum()) == report["n_divergence_flags"],
            "Indikator overfitting berubah")
    require(len(details) == 20 and len(grouped) == 4, "Jumlah model/fold tidak sesuai")
    print(f"VERIFIED | {STATUS} | {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Validasi 20 history dan ringkasan tanpa menulis file")
    group.add_argument("--run", action="store_true", help="Buat ringkasan, kurva, dan laporan audit")
    group.add_argument("--verify", action="store_true", help="Cocokkan laporan dengan hash output & sumber")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.check:
        _, details, grouped, _, _ = collect()
        display(details, grouped)
        print("PREFLIGHT PASS")
    else:
        run()


if __name__ == "__main__":
    main()

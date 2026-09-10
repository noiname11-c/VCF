import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_CLASS_CANDIDATES = ["cls_name", "class", "category", "object", "objects", "cls"]
CSV_SCORE_CANDIDATES = ["score", "anomaly_score", "pr_sp", "pred_score", "image_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot YDFID-1 anomaly score histograms (probability density) for all classes."
    )
    parser.add_argument(
        "--scores-csv",
        type=str,
        default="",
        help="Path to sample-level anomaly score CSV (recommended).",
    )
    parser.add_argument(
        "--npy-root",
        type=str,
        default="results/test_ydfid/npy",
        help="Root folder containing per-image anomaly map npy files grouped by class.",
    )
    parser.add_argument(
        "--class-root",
        type=str,
        default="dataset/mvisa/data/ydfid",
        help="YDFID-1 class root used to enumerate all categories.",
    )
    parser.add_argument(
        "--class-column",
        type=str,
        default="",
        help="Class column in CSV. If empty, auto-detect.",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="",
        help="Score column in CSV. If empty, auto-detect.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for histogram.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=2000,
        help="Use top-k pixels mean as image-level score when reading npy maps.",
    )
    parser.add_argument(
        "--normalize-per-class",
        action="store_true",
        help="Normalize scores to [0,1] inside each class before plotting.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Columns of per-class subplot grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/test_ydfid/score_hist",
        help="Output directory for figures and score summary.",
    )
    return parser.parse_args()


def class_sort_key(name: str):
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def list_classes(class_root: Path) -> list[str]:
    if not class_root.exists():
        return []
    classes = [p.name for p in class_root.iterdir() if p.is_dir()]
    return sorted(classes, key=class_sort_key)


def detect_column(df: pd.DataFrame, user_column: str, candidates: list[str], kind: str) -> str:
    if user_column:
        if user_column not in df.columns:
            raise ValueError(f"{kind} column '{user_column}' not found in CSV columns: {list(df.columns)}")
        return user_column

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"Cannot auto-detect {kind} column. CSV columns: {list(df.columns)}")


def load_scores_from_csv(csv_path: Path, class_column: str, score_column: str) -> dict[str, list[float]]:
    df = pd.read_csv(csv_path)
    cls_col = detect_column(df, class_column, CSV_CLASS_CANDIDATES, "class")
    scr_col = detect_column(df, score_column, CSV_SCORE_CANDIDATES, "score")

    score_map: dict[str, list[float]] = defaultdict(list)
    for _, row in df.iterrows():
        cls_name = str(row[cls_col])
        score = pd.to_numeric(row[scr_col], errors="coerce")
        if pd.isna(score):
            continue
        score_map[cls_name].append(float(score))
    return score_map


def is_valid_map_file(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".npy"):
        return False

    # Keep anomaly map like files and skip side products.
    if name.startswith("gt_") or name.startswith("text_"):
        return False
    if "mask" in name:
        return False
    return True


def map_to_score(arr: np.ndarray, topk: int) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return float("nan")
    if arr.ndim == 0:
        return float(arr)

    flat = arr.reshape(-1)
    if topk <= 0 or topk >= flat.size:
        return float(np.mean(flat))

    pivot = flat.size - topk
    top = np.partition(flat, pivot)[pivot:]
    return float(np.mean(top))


def load_scores_from_npy(npy_root: Path, topk: int) -> dict[str, list[float]]:
    score_map: dict[str, list[float]] = defaultdict(list)
    if not npy_root.exists():
        return score_map

    for cls_dir in sorted([p for p in npy_root.iterdir() if p.is_dir()], key=lambda p: class_sort_key(p.name)):
        for npy_file in cls_dir.rglob("*.npy"):
            if not is_valid_map_file(npy_file):
                continue

            try:
                arr = np.load(npy_file)
            except Exception:
                continue

            score = map_to_score(arr, topk=topk)
            if np.isfinite(score):
                score_map[cls_dir.name].append(score)
    return score_map


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if abs(max_v - min_v) < 1e-12:
        return [0.0 for _ in scores]
    return ((arr - min_v) / (max_v - min_v)).tolist()


def save_summary(score_map: dict[str, list[float]], classes: list[str], output_dir: Path) -> None:
    rows = []
    for cls in classes:
        scores = score_map.get(cls, [])
        if len(scores) == 0:
            rows.append({"class": cls, "count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan})
            continue
        arr = np.asarray(scores, dtype=np.float32)
        rows.append(
            {
                "class": cls,
                "count": int(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "ydfid_score_summary.csv", index=False)


def plot_overlay(score_map: dict[str, list[float]], classes: list[str], bins: int, output_dir: Path) -> None:
    plt.figure(figsize=(14, 8))

    plotted = 0
    for cls in classes:
        scores = score_map.get(cls, [])
        if len(scores) == 0:
            continue
        plt.hist(
            scores,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            alpha=0.95,
            label=f"{cls} (n={len(scores)})",
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No valid scores found. Cannot plot histograms.")

    plt.xlabel("Score range")
    plt.ylabel("Probability density")
    plt.title("YDFID-1 anomaly score histograms (all classes)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "ydfid_hist_all_classes_overlay.png", dpi=200)
    plt.close()


def plot_grid(score_map: dict[str, list[float]], classes: list[str], bins: int, grid_cols: int, output_dir: Path) -> None:
    valid_classes = [cls for cls in classes if len(score_map.get(cls, [])) > 0]
    if not valid_classes:
        raise RuntimeError("No valid class scores found. Cannot plot per-class histograms.")

    cols = max(1, grid_cols)
    rows = math.ceil(len(valid_classes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes_arr = np.array(axes).reshape(-1)

    for ax, cls in zip(axes_arr, valid_classes):
        scores = score_map[cls]
        ax.hist(scores, bins=bins, density=True, color="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.7)
        ax.set_title(f"{cls} (n={len(scores)})", fontsize=10)
        ax.set_xlabel("Score range", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in axes_arr[len(valid_classes):]:
        ax.axis("off")

    fig.suptitle("YDFID-1 anomaly score histograms per class", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "ydfid_hist_per_class_grid.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_root = Path(args.class_root)
    classes = list_classes(class_root)

    if args.scores_csv:
        csv_path = Path(args.scores_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        score_map = load_scores_from_csv(csv_path, args.class_column, args.score_column)
    else:
        npy_root = Path(args.npy_root)
        score_map = load_scores_from_npy(npy_root, topk=args.topk)

    if not classes:
        classes = sorted(score_map.keys(), key=class_sort_key)

    # Ensure unseen classes in score source are still included.
    unseen = [c for c in sorted(score_map.keys(), key=class_sort_key) if c not in classes]
    classes.extend(unseen)

    if args.normalize_per_class:
        for cls in classes:
            if score_map.get(cls):
                score_map[cls] = normalize_scores(score_map[cls])

    save_summary(score_map, classes, output_dir)
    plot_overlay(score_map, classes, bins=args.bins, output_dir=output_dir)
    plot_grid(score_map, classes, bins=args.bins, grid_cols=args.grid_cols, output_dir=output_dir)

    total_samples = sum(len(score_map.get(c, [])) for c in classes)
    valid_classes = sum(1 for c in classes if len(score_map.get(c, [])) > 0)
    print(f"Done. classes_with_scores={valid_classes}, total_samples={total_samples}")
    print(f"Saved figures and summary to: {output_dir}")


if __name__ == "__main__":
    main()

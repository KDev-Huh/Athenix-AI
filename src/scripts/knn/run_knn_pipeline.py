"""
KNN Similar Situation Search Pipeline

Full flow:
    Step 1. VAEPMerger        - StatsBomb events -> VAEP + 360 merge -> CSV
    Step 2. SituationIndex    - CSV -> 50-dim vectors + KNN index
    Step 3. SituationFinder   - Input coordinates -> top-k similar situations
    Step 4. Print results + visualize on soccer pitch

Usage:
    # Run full pipeline (preprocessing included)
    python -m src.scripts.knn.run_knn_pipeline

    # Skip preprocessing if CSV already exists
    python -m src.scripts.knn.run_knn_pipeline --skip-preprocessing
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.knn  import VAEPMerger
from src.recommendation.knn import SituationIndex, SituationFinder

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "data/StatsBombGithub/external/knn/data"
MODEL_DIR   = str(ROOT / "models/vaep")
OUTPUT_CSV  = ROOT / "data/StatsBombGithub/processed/vaep_360_merged.csv"
INDEX_DIR   = str(ROOT / "models/knn_index")
OUT_PNG     = ROOT / "data/StatsBombGithub/processed/knn_result.png"

SCORES_MODEL   = "worldcup_2018_scores.json"
CONCEDES_MODEL = "worldcup_2018_concedes.json"

# competitions.json에 등록된 모든 대회 ID를 동적으로 읽어 전 시즌 포함
import json as _json
with open(DATA_ROOT / "competitions.json", encoding="utf-8") as _f:
    COMPETITIONS = sorted({c["competition_id"] for c in _json.load(_f)})

MAX_DISTANCE = 60
TOP_K        = 5

# Pitch dimensions (StatsBomb coordinate system)
PITCH_LEN = 120
PITCH_WID = 80

# Role color map
COLOR = {1: "#1E90FF", -1: "#FF4444", 2: "#FFD700"}
LABEL = {1: "Teammate", -1: "Opponent", 2: "Goalkeeper"}


# ── Step 1: Preprocess ────────────────────────────────────────────────────────

def step1_preprocess() -> None:
    print("\n[Step 1] Preprocessing: VAEPMerger")
    merger = VAEPMerger(data_root=str(DATA_ROOT), model_dir=MODEL_DIR)
    merger.load_models(SCORES_MODEL, CONCEDES_MODEL)
    merger.run(competitions=COMPETITIONS, output_path=str(OUTPUT_CSV))


# ── Step 2: Build index ───────────────────────────────────────────────────────

def step2_build_index() -> SituationIndex:
    print("\n[Step 2] Building KNN index: SituationIndex")
    idx = SituationIndex()
    idx.build(str(OUTPUT_CSV))
    idx.save(INDEX_DIR)
    print(f"  Index saved: {INDEX_DIR}")
    return idx


# ── Step 3: Query ─────────────────────────────────────────────────────────────

def step3_query(idx: SituationIndex, df: pd.DataFrame):
    """
    Uses the highest-VAEP shot event from the dataset as the query.
    In production, replace players / ball_pos / carrier_pos with
    live output from TopviewAnalyzer.
    """
    print("\n[Step 3] Searching: SituationFinder")

    finder = SituationFinder(idx)

    sample_meta = (
        idx.meta[(idx.meta["type_name"] == "shot") & (idx.meta["result_name"] == "success")]
        .sort_values("vaep_value", ascending=False)
        .iloc[0]
    )
    sample_row  = df[df["event_id"] == sample_meta["event_id"]].iloc[0]
    carrier_pos = (float(sample_row["carrier_x"]), float(sample_row["carrier_y"]))
    ball_pos    = (float(sample_row["start_x"]),   float(sample_row["start_y"]))
    players     = json.loads(sample_row["freeze_frame"])

    print(f"\n  Query event:")
    print(f"    event_id    = {sample_meta['event_id']}")
    print(f"    type        = {sample_meta['type_name']} / {sample_meta['result_name']}")
    print(f"    VAEP        = {sample_meta['vaep_value']:.4f}")
    print(f"    carrier_pos = ({carrier_pos[0]:.1f}, {carrier_pos[1]:.1f})")
    print(f"    ball_pos    = ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
    print(f"    players     = {len(players)}")

    results = finder.find(
        players=players,
        ball_pos=ball_pos,
        carrier_pos=carrier_pos,
        max_distance=MAX_DISTANCE,
        top_k=TOP_K,
    )
    return sample_meta, players, ball_pos, carrier_pos, results


# ── Step 4: Print + Visualize ─────────────────────────────────────────────────

def step4_output(
    sample_meta, players, ball_pos, carrier_pos, results, df: pd.DataFrame
) -> None:
    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n[Step 4] Results  (top-{TOP_K}, max_distance={MAX_DISTANCE})")

    if results.empty:
        print("  No similar situations found within max_distance.")
        return

    print(f"\n  {'Rank':<5} {'Type':<12} {'Result':<10} {'VAEP':>8} {'Dist':>8}  Event ID")
    print("  " + "-" * 74)
    for rank, (_, r) in enumerate(results.iterrows(), start=1):
        print(
            f"  {rank:<5} {r['type_name']:<12} {r['result_name']:<10} "
            f"{r['vaep_value']:>8.4f} {r['distance']:>8.2f}  {r['event_id']}"
        )

    # ── Visualize ─────────────────────────────────────────────────────────────
    n_panels = min(len(results), 2) + 1   # query + up to 2 similar
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    fig.patch.set_facecolor("#1a1a2e")

    # Panel 0 — query only
    _draw_pitch(axes[0])
    _draw_players(axes[0], players, ball_pos, carrier_pos, solid=True)
    axes[0].set_title(
        f"[Query]\n{sample_meta['type_name']} / {sample_meta['result_name']}"
        f"  VAEP={sample_meta['vaep_value']:.3f}",
        color="white", fontsize=10,
    )

    # Panels 1, 2 — query (hollow) + similar event (filled) overlaid
    for i, (_, r) in enumerate(results.head(2).iterrows()):
        ax = axes[i + 1]
        sim_players, sim_ball, sim_carrier = _get_frame(r["event_id"], df)
        _draw_pitch(ax)
        _draw_players(ax, players,     ball_pos,  carrier_pos, solid=False)  # query (hollow)
        _draw_players(ax, sim_players, sim_ball,  sim_carrier, solid=True)   # result (filled)
        ax.set_title(
            f"[Similar #{i + 1}]  dist={r['distance']:.1f}\n"
            f"{r['type_name']} / {r['result_name']}  VAEP={r['vaep_value']:.3f}",
            color="white", fontsize=10,
        )

    # Legend
    legend_items = [
        mpatches.Patch(fc="#1E90FF", label="Teammate"),
        mpatches.Patch(fc="#FF4444", label="Opponent"),
        mpatches.Patch(fc="#FFD700", label="Goalkeeper"),
        plt.scatter([], [], c="white",   s=200, marker="*", label="Carrier"),
        plt.scatter([], [], c="#FFD700", s=80,  marker="o", label="Ball"),
        mpatches.Patch(fc="none", ec="gray", lw=2, label="Query (hollow)"),
        mpatches.Patch(fc="#888888",           label="Result (filled)"),
    ]
    fig.legend(
        handles=legend_items, loc="lower center", ncol=7,
        facecolor="#222", labelcolor="white", fontsize=9, framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"\n  Plot saved: {OUT_PNG}")


# ── Pitch drawing helpers ─────────────────────────────────────────────────────

def _draw_pitch(ax) -> None:
    ax.set_facecolor("#4a7c4e")
    ax.set_xlim(0, PITCH_LEN)
    ax.set_ylim(0, PITCH_WID)
    ax.set_aspect("equal")
    # Center line
    ax.axvline(60, color="white", lw=1, alpha=0.6)
    # Center circle
    ax.add_patch(plt.Circle((60, 40), 9.15, fill=False, ec="white", lw=1, alpha=0.6))
    # Penalty boxes
    for x, y, w, h in [(102, 18, 18, 44), (0, 18, 18, 44)]:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="white", lw=1, alpha=0.6))
    # Goal frames
    for x, y, w, h in [(118, 36, 2, 8), (0, 36, 2, 8)]:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=2, alpha=0.9))


def _draw_players(ax, players, ball_pos, carrier_pos, solid: bool = True) -> None:
    """
    solid=True  — filled circles (retrieved similar event)
    solid=False — hollow circles, semi-transparent (query input)
    """
    alpha = 1.0 if solid else 0.55
    for p in players:
        c = COLOR[p["role"]]
        if solid:
            ax.scatter(p["x"], p["y"], c=c, s=150,
                       edgecolors="white", linewidths=0.7, alpha=alpha, zorder=4)
        else:
            ax.scatter(p["x"], p["y"], c="none", s=200,
                       edgecolors=c, linewidths=2.0, alpha=alpha, zorder=3)

    # Carrier (star)
    ax.scatter(
        *carrier_pos,
        c="white" if solid else "none", s=320, marker="*", zorder=5,
        edgecolors="black" if solid else "white", linewidths=1.2, alpha=alpha,
    )
    # Ball (circle)
    ax.scatter(
        *ball_pos,
        c="#FFD700" if solid else "none", s=100, zorder=5,
        edgecolors="#FFD700", linewidths=1.5, alpha=alpha,
    )


def _get_frame(event_id: str, df: pd.DataFrame):
    """Retrieve players / ball_pos / carrier_pos from the new CSV format."""
    row = df[df["event_id"] == event_id].iloc[0]
    pts = json.loads(row["freeze_frame"])
    return pts, (row["start_x"], row["start_y"]), (row["carrier_x"], row["carrier_y"])


# ── Entry point ───────────────────────────────────────────────────────────────

def main(skip_preprocessing: bool = False) -> None:
    print("=" * 60)
    print("  KNN Similar Situation Search Pipeline")
    print("=" * 60)

    if skip_preprocessing:
        if not OUTPUT_CSV.exists():
            print(f"\n[Error] CSV not found: {OUTPUT_CSV}")
            print("  Run without --skip-preprocessing first.")
            sys.exit(1)
        print(f"\n[Step 1] Skipped (CSV exists: {OUTPUT_CSV})")
    else:
        step1_preprocess()

    idx = step2_build_index()

    print("\n  Loading CSV for query construction...")
    df = pd.read_csv(OUTPUT_CSV)

    sample_meta, players, ball_pos, carrier_pos, results = step3_query(idx, df)
    step4_output(sample_meta, players, ball_pos, carrier_pos, results, df)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Similar Situation Search Pipeline")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip Step 1 if vaep_360_merged.csv already exists",
    )
    args = parser.parse_args()
    main(skip_preprocessing=args.skip_preprocessing)

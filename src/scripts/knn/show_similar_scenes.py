"""
Show Similar Scenes

Picks any event from the full dataset as a query and finds the
2 most similar situations across ALL 118,555 events (no type
filtering, no distance threshold).

Sorted purely by euclidean distance — most similar first.

Usage:
    # Random query event
    python -m src.scripts.knn.show_similar_scenes

    # Specific event ID as query
    python -m src.scripts.knn.show_similar_scenes --event-id <uuid>
"""

import argparse
import json
import random
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

from src.recommendation.knn import SituationIndex, SituationFinder

OUTPUT_CSV = ROOT / "data/StatsBombGithub/processed/vaep_360_merged.csv"
INDEX_DIR  = str(ROOT / "models/knn_index")
OUT_PNG    = ROOT / "data/StatsBombGithub/processed/similar_scenes.png"

COLOR = {1: "#1E90FF", -1: "#FF4444", 2: "#FFD700"}
BG    = "#1a1a2e"


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index() -> SituationIndex:
    idx    = SituationIndex()
    nn_pkl = Path(INDEX_DIR) / "nn.pkl"
    if nn_pkl.exists():
        print(f"Loading index from {INDEX_DIR} ...")
        idx.load(INDEX_DIR)
    else:
        print(f"Building index from {OUTPUT_CSV} ...")
        idx.build(str(OUTPUT_CSV))
    return idx


# ── Search ────────────────────────────────────────────────────────────────────

def get_top2_by_distance(idx, finder, players, ball_pos, carrier_pos, query_event_id):
    """
    Search ALL events in the index.
    Returns the 2 closest events, excluding the query itself.
    No distance threshold — purely nearest neighbours.
    """
    q         = finder._vectorize_query(players, ball_pos, carrier_pos)
    # Fetch a few extra in case the query event itself is in the index
    dists, js = idx.kneighbors(q, n_neighbors=5)
    dists, js = dists[0], js[0]

    result             = idx.meta.iloc[js].copy()
    result["distance"] = dists

    # Exclude the query event itself
    result = (result[result["event_id"] != query_event_id]
              .sort_values("distance", ascending=True)
              .head(2)
              .reset_index(drop=True))
    return result


# ── Pitch drawing ─────────────────────────────────────────────────────────────

def draw_pitch(ax):
    ax.set_facecolor("#4a7c4e")
    ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect("equal")
    ax.axvline(60, color="white", lw=1, alpha=0.6)
    ax.add_patch(plt.Circle((60, 40), 9.15, fill=False, ec="white", lw=1, alpha=0.6))
    for x, y, w, h in [(102, 18, 18, 44), (0, 18, 18, 44)]:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="white", lw=1, alpha=0.6))
    for x, y, w, h in [(118, 36, 2, 8), (0, 36, 2, 8)]:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=2, alpha=0.9))


def draw_players(ax, players, ball_pos, carrier_pos, solid=True):
    alpha = 1.0 if solid else 0.50
    for p in players:
        c = COLOR[p["role"]]
        if solid:
            ax.scatter(p["x"], p["y"], c=c, s=150,
                       edgecolors="white", linewidths=0.7, alpha=alpha, zorder=4)
        else:
            ax.scatter(p["x"], p["y"], c="none", s=210,
                       edgecolors=c, linewidths=2.2, alpha=alpha, zorder=3)
    ax.scatter(*carrier_pos,
               c="white" if solid else "none", s=320, marker="*", zorder=5,
               edgecolors="black" if solid else "white", linewidths=1.2, alpha=alpha)
    ax.scatter(*ball_pos,
               c="#FFD700" if solid else "none", s=100, zorder=5,
               edgecolors="#FFD700", linewidths=1.5, alpha=alpha)


def get_frame(event_id, df):
    row = df[df["event_id"] == event_id].iloc[0]
    return (
        json.loads(row["freeze_frame"]),
        (float(row["start_x"]),   float(row["start_y"])),
        (float(row["carrier_x"]), float(row["carrier_y"])),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(event_id: str = None):
    idx    = load_index()
    finder = SituationFinder(idx)

    print("Loading CSV ...")
    df = pd.read_csv(OUTPUT_CSV)

    # ── Query selection ───────────────────────────────────────────────────────
    if event_id:
        if event_id not in idx.meta["event_id"].values:
            print(f"[Error] event_id not found: {event_id}")
            sys.exit(1)
        query_meta = idx.meta[idx.meta["event_id"] == event_id].iloc[0]
    else:
        # Random event — no type restriction
        query_meta = idx.meta.sample(1, random_state=random.randint(0, 9999)).iloc[0]

    query_row   = df[df["event_id"] == query_meta["event_id"]].iloc[0]
    carrier_pos = (float(query_row["carrier_x"]), float(query_row["carrier_y"]))
    ball_pos    = (float(query_row["start_x"]),   float(query_row["start_y"]))
    players     = json.loads(query_row["freeze_frame"])

    print(f"\nQuery event  (total index size: {len(idx.meta):,} events):")
    print(f"  event_id = {query_meta['event_id']}")
    print(f"  type     = {query_meta['type_name']} / {query_meta['result_name']}")
    print(f"  VAEP     = {query_meta['vaep_value']:.4f}")
    print(f"  carrier  = ({carrier_pos[0]:.1f}, {carrier_pos[1]:.1f})")
    print(f"  ball     = ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
    print(f"  players  = {len(players)}")

    # ── Search ALL events, no filter ──────────────────────────────────────────
    results = get_top2_by_distance(
        idx, finder, players, ball_pos, carrier_pos,
        query_event_id=query_meta["event_id"],
    )

    print(f"\nTop-2 most similar situations (from all {len(idx.meta):,} events):\n")
    print(f"  {'Rank':<5} {'Dist':>7} {'VAEP':>8}  {'Type':<12} {'Result'}")
    print("  " + "-" * 52)
    for rank, (_, r) in enumerate(results.iterrows(), start=1):
        print(f"  {rank:<5} {r['distance']:>7.2f} {r['vaep_value']:>8.4f}"
              f"  {r['type_name']:<12} {r['result_name']}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor(BG)

    # Panel 0 — query alone
    draw_pitch(axes[0])
    draw_players(axes[0], players, ball_pos, carrier_pos, solid=True)
    axes[0].set_title(
        f"[Query]\n{query_meta['type_name']} / {query_meta['result_name']}"
        f"\nVAEP = {query_meta['vaep_value']:.3f}",
        color="white", fontsize=11,
    )

    # Panels 1-2 — similar scenes overlaid
    for rank, (_, r) in enumerate(results.iterrows(), start=1):
        ax = axes[rank]
        sim_players, sim_ball, sim_carrier = get_frame(r["event_id"], df)
        draw_pitch(ax)
        draw_players(ax, players,     ball_pos,  carrier_pos, solid=False)
        draw_players(ax, sim_players, sim_ball,  sim_carrier, solid=True)
        ax.set_title(
            f"[Similar #{rank}]  dist = {r['distance']:.2f}\n"
            f"{r['type_name']} / {r['result_name']}"
            f"\nVAEP = {r['vaep_value']:.3f}",
            color="white", fontsize=11,
        )

    legend_items = [
        mpatches.Patch(fc="#1E90FF", label="Teammate"),
        mpatches.Patch(fc="#FF4444", label="Opponent"),
        mpatches.Patch(fc="#FFD700", label="Goalkeeper"),
        plt.scatter([], [], c="white",   s=200, marker="*", label="Carrier"),
        plt.scatter([], [], c="#FFD700", s=80,  marker="o", label="Ball"),
        mpatches.Patch(fc="none", ec="gray", lw=2, label="Query (hollow)"),
        mpatches.Patch(fc="#888888",           label="Result (filled)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=7,
               facecolor="#222", labelcolor="white", fontsize=9, framealpha=0.9)
    plt.suptitle(
        f"Top-2 Most Similar Situations  —  searched across all {len(idx.meta):,} events",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"\nPlot saved: {OUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top-2 most similar scenes from all events")
    parser.add_argument(
        "--event-id", type=str, default=None,
        help="Query event UUID. If omitted, picks a random event from the dataset.",
    )
    args = parser.parse_args()
    main(event_id=args.event_id)

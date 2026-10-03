"""
Data cleaning script for the Elliptic Bitcoin Dataset.

NOTE: Unknown-labeled nodes are RETAINED in all outputs so that GNN models
(GCN, GAT) can propagate messages through them during training. Supervised
training and evaluation use a boolean mask that includes only the ~46,564
labeled nodes (illicit + licit), ensuring the model learns and is evaluated
only on verified labels while preserving the full graph topology.

Usage:
    python scripts/clean_data.py

Input (place in ./data/):
    - elliptic_txs_classes.csv
    - elliptic_txs_edgelist.csv
    - elliptic_txs_features.csv

Output (written to ./data/):
    - processed_classifications.csv (all nodes, labels recoded to 0/1/-1)
    - processed_edges.csv           (all edges, unchanged)
    - processed_features.csv        (all nodes, columns renamed)
"""

import pandas as pd
from pathlib import Path


DATA_DIR = Path("./data")


def process_classes(path: Path) -> pd.DataFrame:
    """
    Load class labels and recode them.
    Original: '1' = illicit, '2' = licit, 'unknown' = unlabeled
    Recoded:   1  = illicit,  0  = licit,  -1       = unknown
    """
    class_df = pd.read_csv(path)

    # Recode labels: illicit=1, licit=0, unknown=-1
    label_map = {"1": 1, "2": 0, "unknown": -1}
    class_df["class"] = class_df["class"].map(label_map).astype(int)

    total = len(class_df)
    illicit = (class_df["class"] == 1).sum()
    licit = (class_df["class"] == 0).sum()
    unknown = (class_df["class"] == -1).sum()

    print(f"Classes — total: {total:,}")
    print(f"  illicit (1):   {illicit:,} ({100*illicit/total:.2f}%)")
    print(f"  licit   (0):   {licit:,} ({100*licit/total:.2f}%)")
    print(f"  unknown (-1):  {unknown:,} ({100*unknown/total:.2f}%)")

    return class_df


def process_edges(path: Path) -> pd.DataFrame:
    """Load the edge list unchanged. All edges are retained for the graph."""
    edge_df = pd.read_csv(path)
    print(f"Edges   — total: {len(edge_df):,} (all retained)")
    return edge_df


def process_features(path: Path) -> pd.DataFrame:
    """Load features and assign readable column headers."""
    features_df = pd.read_csv(path, header=None)

    # The Elliptic dataset has 166 numerical columns after txId and time_step:
    # 94 local features + 72 aggregated one-hop neighborhood features
    n_feat = features_df.shape[1] - 2
    headers = ["txId", "time_step"] + [f"feat_{i + 1}" for i in range(n_feat)]
    features_df.columns = headers

    print(f"Features — total: {len(features_df):,} (all retained)")
    return features_df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Processing Elliptic Bitcoin Dataset")
    print("Unknown nodes are RETAINED for GNN message passing")
    print("=" * 60)

    # Step 1: process class labels (recode but keep all rows)
    class_df = process_classes(DATA_DIR / "elliptic_txs_classes.csv")

    # Step 2: process edges and features (all retained)
    edge_df = process_edges(DATA_DIR / "elliptic_txs_edgelist.csv")
    features_df = process_features(DATA_DIR / "elliptic_txs_features.csv")

    # Step 3: save processed files (no pandas index column written)
    class_df.to_csv(DATA_DIR / "processed_classifications.csv", index=False)
    edge_df.to_csv(DATA_DIR / "processed_edges.csv", index=False)
    features_df.to_csv(DATA_DIR / "processed_features.csv", index=False)

    print("=" * 60)
    print(f"Done. Processed files written to {DATA_DIR.resolve()}")
    print()
    print("For GNN training, use a boolean mask on the labels column:")
    print("    train_mask = class_df['class'] != -1  # ~46,564 labeled nodes")
    print("=" * 60)


if __name__ == "__main__":
    main()
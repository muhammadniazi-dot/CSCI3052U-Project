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

REQUIRED_FILES = [
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_features.csv",
]


def check_required_files() -> None:
    """Fail early with a helpful message if the raw CSVs aren't present."""
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {DATA_DIR.resolve()}: {missing}. "
            "Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set "
            "and place the CSVs in the ./data/ directory."
        )


def process_classes(path: Path) -> pd.DataFrame:
    """
    Load class labels and recode them.
    Original: '1' = illicit, '2' = licit, 'unknown' = unlabeled
    Recoded:   1  = illicit,  0  = licit,  -1       = unknown
    """
    class_df = pd.read_csv(path)

    # Recode labels: illicit=1, licit=0, unknown=-1
    label_map = {"1": 1, "2": 0, "unknown": -1}
    class_df["class"] = class_df["class"].astype(str).map(label_map)
    assert class_df["class"].notna().all(), (
        "Unexpected label values found. Expected '1', '2', or 'unknown'."
    )
    class_df["class"] = class_df["class"].astype(int)

    total = len(class_df)
    illicit = (class_df["class"] == 1).sum()
    licit = (class_df["class"] == 0).sum()
    unknown = (class_df["class"] == -1).sum()

    print(f"Classes  — total: {total:,}")
    print(f"  illicit (1):   {illicit:,} ({100*illicit/total:.2f}%)")
    print(f"  licit   (0):   {licit:,} ({100*licit/total:.2f}%)")
    print(f"  unknown (-1):  {unknown:,} ({100*unknown/total:.2f}%)")

    return class_df


def process_edges(path: Path) -> pd.DataFrame:
    """Load the edge list unchanged. All edges are retained for the graph."""
    edge_df = pd.read_csv(path)
    print(f"Edges    — total: {len(edge_df):,} (all retained)")
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


def validate_data(
    class_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
    """Verify data integrity assumptions documented in the Milestone 2 memo."""
    # No duplicate transaction IDs
    assert class_df["txId"].is_unique, "Duplicate txIds in classes CSV"
    assert features_df["txId"].is_unique, "Duplicate txIds in features CSV"

    # No missing values in features
    n_missing = features_df.isnull().sum().sum()
    assert n_missing == 0, f"Found {n_missing} missing feature values"

    # No self-loops in edges
    self_loops = (edge_df["txId1"] == edge_df["txId2"]).sum()
    assert self_loops == 0, f"Found {self_loops} self-loop edges"

    # Every edge references a valid txId
    valid_ids = set(features_df["txId"])
    edge_ids = set(edge_df["txId1"]) | set(edge_df["txId2"])
    orphan_edges = edge_ids - valid_ids
    assert len(orphan_edges) == 0, (
        f"Found {len(orphan_edges)} edges referencing missing txIds"
    )

    print("Data integrity checks passed")


def save_outputs(
    class_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
    """Write the processed CSVs and report their file sizes."""
    outputs = {
        "processed_classifications.csv": class_df,
        "processed_edges.csv": edge_df,
        "processed_features.csv": features_df,
    }
    for filename, df in outputs.items():
        out_path = DATA_DIR / filename
        df.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {filename}: {size_mb:.1f} MB")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Processing Elliptic Bitcoin Dataset")
    print("Unknown nodes are RETAINED for GNN message passing")
    print("=" * 60)

    # Step 0: fail fast if any raw files are missing
    check_required_files()

    # Step 1: load and process each file
    class_df = process_classes(DATA_DIR / "elliptic_txs_classes.csv")
    edge_df = process_edges(DATA_DIR / "elliptic_txs_edgelist.csv")
    features_df = process_features(DATA_DIR / "elliptic_txs_features.csv")

    # Step 2: verify data integrity assumptions
    print()
    validate_data(class_df, edge_df, features_df)

    # Step 3: save processed outputs
    print()
    print(f"Writing processed files to {DATA_DIR.resolve()}:")
    save_outputs(class_df, edge_df, features_df)

    print("=" * 60)
    print("Done.")
    print()
    print("For GNN training, use a boolean mask on the labels column:")
    print("    train_mask = class_df['class'] != -1  # ~46,564 labeled nodes")
    print("=" * 60)


if __name__ == "__main__":
    main()
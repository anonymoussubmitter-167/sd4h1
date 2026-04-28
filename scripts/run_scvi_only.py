"""Run scVI baseline only and merge into existing chisel_benchmark.json."""
import sys, json, argparse
sys.path.insert(0, "/home/bcheng/ROSETTA")
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()

    from scripts.benchmark_chisel import (
        get_scvi_embeddings, get_chisel_embeddings, get_labels,
        compute_clustering_metrics, morans_i, PROCESSED_DIR, PROJECT_ROOT
    )
    import anndata as ad
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    output_dir = PROJECT_ROOT / "outputs" / args.dataset / "chisel_benchmark"
    out_file = output_dir / "chisel_benchmark.json"

    # Load existing results
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        logger.info("Loaded existing results with %d keys", len(existing))

    # Get CHISEL embeddings for spatial coords and labels
    logger.info("Loading CHISEL embeddings for spatial coords...")
    z_chisel, spatial_coords, expr_matrix = get_chisel_embeddings(args.dataset, args.max_nodes)
    adata = ad.read_h5ad(PROCESSED_DIR / f"{args.dataset}.h5ad")
    labels = get_labels(adata, args.max_nodes)
    labels_trimmed = {k: v[:z_chisel.shape[0]] for k, v in labels.items()}

    # Run scVI
    logger.info("=== scVI (max_epochs=%d) ===", args.max_epochs)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
    z_scvi = get_scvi_embeddings(args.dataset, args.max_nodes, max_epochs=args.max_epochs)
    n_scvi = min(z_scvi.shape[0], z_chisel.shape[0])
    labels_scvi = {k: v[:n_scvi] for k, v in labels_trimmed.items()}
    scvi_metrics = compute_clustering_metrics(z_scvi[:n_scvi], labels_scvi)
    mi_scvi = morans_i(z_scvi[:n_scvi], spatial_coords[:n_scvi], k=6)
    scvi_metrics["morans_i"] = mi_scvi
    for k, v in sorted(scvi_metrics.items()):
        logger.info("  scVI %s: %.4f", k, v)
        existing[f"scvi_{k}"] = v

    # Save merged results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Results saved to %s", out_file)

if __name__ == "__main__":
    main()

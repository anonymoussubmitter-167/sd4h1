"""GO enrichment analysis for COMPASS gene modules."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def module_go_enrichment(
    H: NDArray,
    gene_names: list[str],
    top_n: int = 50,
    organism: str = "human",
    gene_sets: str = "GO_Biological_Process_2023",
) -> list[pd.DataFrame]:
    """Run GO enrichment on top genes from each NMF module.

    Uses gseapy's enrichr interface to query Enrichr gene set libraries.

    Args:
        H: (k, G) gene loadings matrix.
        gene_names: List of G gene names.
        top_n: Number of top genes per module to use for enrichment.
        organism: Organism for Enrichr ('human' or 'mouse').
        gene_sets: Enrichr gene set library name.

    Returns:
        List of DataFrames (one per module) with enrichment results.
        Each DataFrame has columns: Term, Overlap, P-value, Adjusted P-value,
        Genes, Combined Score.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed, skipping GO enrichment")
        return [pd.DataFrame() for _ in range(H.shape[0])]

    results = []
    for m in range(H.shape[0]):
        loadings = H[m]
        top_idx = np.argsort(loadings)[::-1][:top_n]
        top_genes = [gene_names[i] for i in top_idx]

        logger.info("Module %d: enriching %d genes...", m, len(top_genes))

        try:
            enr = gp.enrichr(
                gene_list=top_genes,
                gene_sets=gene_sets,
                organism=organism,
                outdir=None,  # don't write files
                no_plot=True,
                verbose=False,
            )
            df = enr.results
            # Keep significant results
            df = df[df["Adjusted P-value"] < 0.05].head(20)
            results.append(df)
            if len(df) > 0:
                logger.info("  Module %d: %d significant GO terms (top: %s)",
                            m, len(df), df.iloc[0]["Term"] if len(df) > 0 else "none")
            else:
                logger.info("  Module %d: no significant GO terms", m)
        except Exception as e:
            logger.warning("  Module %d enrichment failed: %s", m, e)
            results.append(pd.DataFrame())

    return results


def module_go_enrichment_prerank(
    H: NDArray,
    gene_names: list[str],
    organism: str = "human",
    gene_sets: str = "GO_Biological_Process_2023",
) -> list[pd.DataFrame]:
    """Run preranked GSEA on module gene loadings.

    Instead of using a cutoff, uses all genes ranked by loading weight.

    Args:
        H: (k, G) gene loadings matrix.
        gene_names: List of G gene names.
        organism: Organism for gseapy.
        gene_sets: Gene set library name.

    Returns:
        List of DataFrames (one per module) with GSEA results.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed, skipping prerank enrichment")
        return [pd.DataFrame() for _ in range(H.shape[0])]

    results = []
    for m in range(H.shape[0]):
        # Create ranked gene list (gene -> loading weight)
        rnk = pd.Series(H[m], index=gene_names).sort_values(ascending=False)

        try:
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=gene_sets,
                outdir=None,
                no_plot=True,
                verbose=False,
                min_size=5,
                max_size=500,
                seed=42,
            )
            df = pre_res.res2d
            df = df[df["FDR q-val"] < 0.25].head(20)
            results.append(df)
        except Exception as e:
            logger.warning("Module %d prerank failed: %s", m, e)
            results.append(pd.DataFrame())

    return results

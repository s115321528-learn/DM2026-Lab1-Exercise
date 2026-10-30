"""tools_reduction.py

Dimensionality reduction and label binarization tools for text analysis.
"""

from typing import Literal, Optional, Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from langchain_core.tools import tool


def make_tools(session):
    """Factory function creating reduction and binarization tools bound to the session state."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: Optional[float] = 30.0,
        n_neighbors: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """Reduce dimensionality of the feature matrix using PCA, t-SNE, or UMAP, and generate a scatter plot.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap'). Default is 'pca'.
            n_components: Number of dimensions to reduce to (must be 2). Default is 2.
            random_state: Random seed for reproducibility. Default is 42.
            perplexity: Perplexity parameter for t-SNE (must be positive and < n_samples). Default is 30.0.
            n_neighbors: Number of neighbors parameter for UMAP (must be >= 2 and < n_samples). Default is 15.

        Returns:
            Dict containing result_id, method, coordinates, n_samples, n_components, and extra metadata.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_samples, n_features = X.shape

        if n_components != 2:
            return {"error": f"n_components must be 2, got {n_components}."}

        if n_samples < 2:
            return {"error": f"Insufficient samples for dimensionality reduction: {n_samples} (need at least 2)."}

        # Validate presence and alignment of category labels for plotting
        labels = None
        if session.labels is not None and len(session.labels) == n_samples:
            labels = session.labels
        elif session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].values
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].values

        if labels is None or len(labels) != n_samples:
            return {"error": f"Missing or misaligned document category labels. Found {len(labels) if labels is not None else 0} labels for {n_samples} samples."}

        # Parameter validation per method
        if method == "pca":
            max_components = min(n_samples, n_features)
            if n_components > max_components:
                return {"error": f"n_components ({n_components}) cannot exceed min(n_samples, n_features) ({max_components})."}
        elif method == "tsne":
            if perplexity is None or perplexity <= 0:
                return {"error": f"perplexity must be positive, got {perplexity}."}
            if perplexity >= n_samples:
                return {"error": f"perplexity ({perplexity}) must be strictly less than n_samples ({n_samples})."}
        elif method == "umap":
            if n_neighbors is None or n_neighbors < 2:
                return {"error": f"n_neighbors must be at least 2, got {n_neighbors}."}
            if n_neighbors >= n_samples:
                return {"error": f"n_neighbors ({n_neighbors}) must be strictly less than n_samples ({n_samples})."}
        else:
            return {"error": f"Unsupported method: {method}"}

        # Memory safety check for sparse-to-dense conversion
        if sp.issparse(X):
            estimated_bytes = n_samples * n_features * 8
            if estimated_bytes > 1_000_000_000:  # 1GB limit
                return {"error": f"Dense matrix conversion requires approx {estimated_bytes / (1024**2):.1f} MB, exceeding safety limit."}
            X_dense = X.toarray()
        else:
            X_dense = X

        explained_variance_ratio: Optional[List[float]] = None
        params_used: Dict[str, Any] = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
        }

        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(X_dense if sp.issparse(X) else X)
            if hasattr(pca, "explained_variance_ratio_"):
                explained_variance_ratio = pca.explained_variance_ratio_.tolist()

        elif method == "tsne":
            from sklearn.manifold import TSNE
            params_used["perplexity"] = perplexity
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="random",
            )
            coords = tsne.fit_transform(X_dense)

        elif method == "umap":
            import umap
            params_used["n_neighbors"] = n_neighbors
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)

        # Store artifacts
        session.artifacts["reduced_coordinates"] = coords.tolist()

        # Generate scatter plot colored by valid category labels
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=str(lbl),
                alpha=0.7,
                edgecolors="k"
            )
        ax.legend(title="Category")
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        res_id = session.next_result_id(f"reduce_dim_{method}")
        summary = {
            "result_id": res_id,
            "method": method,
            "n_samples": n_samples,
            "n_components": n_components,
            "coordinates": coords.tolist(),
            "params_used": params_used,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", params_used, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes document category labels using sklearn OneHotEncoder.

        Returns:
            Dict containing result_id, n_samples, categories, and binarized_labels matrix.
        """
        # 1. Label Alignment Check if both session.labels and session.dataframe exist
        if session.labels is not None and len(session.labels) > 0 and session.dataframe is not None and not session.dataframe.empty:
            if len(session.labels) != len(session.dataframe):
                return {
                    "error": f"Label misalignment: session.labels has {len(session.labels)} elements, but session.dataframe has {len(session.dataframe)} rows."
                }

        # 2. Extract labels with "label" column fallback
        labels = None
        if session.labels is not None and len(session.labels) > 0:
            labels = list(session.labels)
        elif session.dataframe is not None and not session.dataframe.empty:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()
            elif "label" in session.dataframe.columns:
                labels = session.dataframe["label"].tolist()

        if labels is None or len(labels) == 0:
            return {"error": "No category labels found in session. Load a dataset or set labels first."}

        # 3. Check for missing/null/NaN category labels
        has_missing = any(pd.isna(lbl) or str(lbl).strip().lower() in ("nan", "none", "null", "") for lbl in labels)
        if has_missing:
            return {"error": "Category labels contain missing, null, or empty values."}

        # 4. Convert labels to 2D numpy array and OneHotEncode with version fallback
        labels_arr = np.array(labels).reshape(-1, 1)

        try:
            encoder = OneHotEncoder(sparse_output=False, dtype=int)
        except TypeError:
            encoder = OneHotEncoder(sparse=False, dtype=int)

        one_hot_matrix = encoder.fit_transform(labels_arr).tolist()
        categories = [str(c) for c in encoder.categories_[0].tolist()]

        n_samples = len(labels)

        # 5. Store in session.artifacts without modifying session.labels, session.categories, or session.dataframe
        session.artifacts["binarized_labels"] = one_hot_matrix
        session.artifacts["binarized_label_classes"] = categories

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "n_samples": n_samples,
            "categories": categories,
            "binarized_labels": one_hot_matrix,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]

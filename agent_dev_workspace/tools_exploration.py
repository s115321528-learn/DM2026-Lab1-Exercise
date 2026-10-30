import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute feature correlation matrix for top variance terms and plot heatmap.

        Args:
            top_n: Number of top terms by variance to select (default 20).

        Returns:
            A summary dictionary containing result_id, top_n, selected_terms,
            and correlation_matrix (as a nested list).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix missing. Run build_dtm_tool first."
            }

        if not isinstance(top_n, int) or top_n <= 0:
            return {"error": "top_n must be a positive integer."}

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_docs = matrix.shape[0]

        if n_docs < 2:
            return {"error": "At least 2 documents are required to compute correlation."}

        if sparse.issparse(matrix):
            mean = np.array(matrix.mean(axis=0)).ravel()
            mean_sq = np.array(matrix.power(2).mean(axis=0)).ravel()
            variances = mean_sq - mean**2
        else:
            variances = np.var(matrix, axis=0)

        k = min(top_n, len(feature_names))
        if k == 0:
            return {"error": "No features available to compute correlation matrix."}

        # Sort terms by variance descending using stable sort to maintain order
        sorted_indices = np.argsort(-variances, kind="stable")
        top_indices = sorted_indices[:k]

        selected_terms = feature_names[top_indices].tolist()

        # Densify only selected top_indices columns
        if sparse.issparse(matrix):
            X_sub = matrix[:, top_indices].toarray()
        else:
            X_sub = np.asarray(matrix)[:, top_indices]

        if k == 1:
            var_single = variances[top_indices[0]]
            corr_matrix = np.array([[1.0 if var_single > 0 else 0.0]])
        else:
            corr_matrix = np.corrcoef(X_sub.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # Build figure and set pending_figure
        fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, k * 0.6)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            cmap="coolwarm",
            ax=ax,
        )
        ax.set_title(f"Top-{k} Term Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        corr_list = corr_matrix.tolist()
        df_report = pd.DataFrame(corr_matrix, index=selected_terms, columns=selected_terms)

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "top_n": top_n,
            "selected_terms": selected_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=df_report,
        )

        return summary

    @tool
    def cosine_similarity_tool(doc_idx_1: int, doc_idx_2: int) -> dict:
        """Compute cosine similarity between two documents in the document-term matrix.

        Args:
            doc_idx_1: Row index of the first document.
            doc_idx_2: Row index of the second document.

        Returns:
            A summary dictionary containing result_id, doc_idx_1, doc_idx_2,
            and similarity (float).
        """
        if session.feature_matrix is None:
            return {
                "error": "Document-term matrix missing. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix
        n_docs = matrix.shape[0]

        if not (0 <= doc_idx_1 < n_docs) or not (0 <= doc_idx_2 < n_docs):
            return {
                "error": f"Document index out of range [0, {n_docs - 1}]."
            }

        if sparse.issparse(matrix):
            v1 = matrix[doc_idx_1].toarray().ravel()
            v2 = matrix[doc_idx_2].toarray().ravel()
        else:
            v1 = np.asarray(matrix[doc_idx_1]).ravel()
            v2 = np.asarray(matrix[doc_idx_2]).ravel()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = float(np.dot(v1, v2) / (norm1 * norm2))
            sim = float(np.nan_to_num(sim, nan=0.0))

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_idx_1": doc_idx_1,
            "doc_idx_2": doc_idx_2,
            "similarity": float(np.round(sim, 6)),
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_idx_1": doc_idx_1, "doc_idx_2": doc_idx_2},
            summary=summary,
        )

        return summary

    return [feature_correlation_matrix_tool, cosine_similarity_tool]

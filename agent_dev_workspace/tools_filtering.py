import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter terms in the document-term matrix based on feature variance.

        Identifies near-constant terms whose variance across documents is below
        the specified threshold.

        Args:
            threshold: Minimum variance required to keep a term. Default is 0.0.

        Returns:
            A summary dictionary with result_id, kept_terms, removed_terms,
            n_kept, and n_removed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix missing. Run build_dtm_tool first."
            }

        if not np.isfinite(threshold) or threshold < 0:
            return {
                "error": "Threshold must be a finite, non-negative float."
            }

        matrix = session.feature_matrix
        if sparse.issparse(matrix):
            # Population variance for sparse matrices without dense conversion: E[X^2] - (E[X])^2
            mean = np.array(matrix.mean(axis=0)).ravel()
            mean_sq = np.array(matrix.power(2).mean(axis=0)).ravel()
            variances = mean_sq - mean**2
        else:
            variances = np.var(matrix, axis=0)

        df = pd.DataFrame(
            {"term": session.feature_names, "variance": variances}
        )
        df = df.sort_values(by="variance", ascending=False).reset_index(
            drop=True
        )

        kept_terms = df[df["variance"] >= threshold]["term"].tolist()
        removed_terms = df[df["variance"] < threshold]["term"].tolist()

        result_id = session.next_result_id("variance")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df,
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter terms based on Pearson correlation with a target class.

        Args:
            target_class: The category name to correlate terms against.
            threshold: Minimum absolute correlation required to keep a term.

        Returns:
            A summary dictionary with result_id, target_class, threshold,
            kept_terms, removed_terms, n_kept, and n_removed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix missing. Run build_dtm_tool first."
            }

        if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
            return {
                "error": "Threshold must be a finite float between 0.0 and 1.0."
            }

        if session.labels is None:
            return {
                "error": "Labels missing from session."
            }

        if target_class not in session.categories:
            return {
                "error": f"Target class '{target_class}' not found in categories."
            }

        y = (np.array(session.labels) == target_class).astype(float)
        y_mean = y.mean()
        y_std = y.std()

        if y_std == 0:
            return {"error": "Target class indicator has zero variance."}

        matrix = session.feature_matrix
        if sparse.issparse(matrix):
            X_mean = np.array(matrix.mean(axis=0)).ravel()
            X_std = np.sqrt(
                np.array(matrix.power(2).mean(axis=0)).ravel() - X_mean**2
            )
            E_Xy = np.array(matrix.T.dot(y)).ravel() / matrix.shape[0]
            cov = E_Xy - X_mean * y_mean
            with np.errstate(divide="ignore", invalid="ignore"):
                r = cov / (X_std * y_std)
                r = np.nan_to_num(r, nan=0.0)
        else:
            r = np.corrcoef(matrix.T, y)[:-1, -1]
            r = np.nan_to_num(r, nan=0.0)

        df = pd.DataFrame({"term": session.feature_names, "pearson_r": r})
        df["abs_r"] = df["pearson_r"].abs()
        df = df.sort_values(by="abs_r", ascending=False).reset_index(
            drop=True
        )

        kept_terms = df[df["abs_r"] >= threshold]["term"].tolist()
        removed_terms = df[df["abs_r"] < threshold]["term"].tolist()
        df = df.drop(columns=["abs_r"])

        result_id = session.next_result_id("pearson")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df,
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter terms based on Spearman rank correlation with a target class.

        Args:
            target_class: The category name to correlate terms against.
            threshold: Minimum absolute correlation required to keep a term.

        Returns:
            A summary dictionary with result_id, target_class, threshold,
            kept_terms, removed_terms, n_kept, and n_removed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix missing. Run build_dtm_tool first."
            }

        if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
            return {
                "error": "Threshold must be a finite float between 0.0 and 1.0."
            }

        if session.labels is None:
            return {
                "error": "Labels missing from session."
            }

        if target_class not in session.categories:
            return {
                "error": f"Target class '{target_class}' not found in categories."
            }

        y = (np.array(session.labels) == target_class).astype(float)
        if y.std() == 0:
            return {"error": "Target class indicator has zero variance."}

        matrix = session.feature_matrix
        n_terms = len(session.feature_names)
        r = np.zeros(n_terms, dtype=float)

        if sparse.issparse(matrix):
            matrix_csc = matrix.tocsc()
            for col_idx in range(n_terms):
                x_col = matrix_csc[:, col_idx].toarray().ravel()
                if np.all(x_col == x_col[0]):
                    res_r = 0.0
                else:
                    res_r, _ = spearmanr(x_col, y)
                r[col_idx] = 0.0 if np.isnan(res_r) else res_r
        else:
            for col_idx in range(n_terms):
                x_col = matrix[:, col_idx]
                if np.all(x_col == x_col[0]):
                    res_r = 0.0
                else:
                    res_r, _ = spearmanr(x_col, y)
                r[col_idx] = 0.0 if np.isnan(res_r) else res_r

        df = pd.DataFrame({"term": session.feature_names, "spearman_r": r})
        df["abs_r"] = df["spearman_r"].abs()
        df = df.sort_values(by="abs_r", ascending=False).reset_index(
            drop=True
        )

        kept_terms = df[df["abs_r"] >= threshold]["term"].tolist()
        removed_terms = df[df["abs_r"] < threshold]["term"].tolist()
        df = df.drop(columns=["abs_r"])

        result_id = session.next_result_id("spearman")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]

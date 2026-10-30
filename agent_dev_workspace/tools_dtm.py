from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from raw text in session.dataframe.

        Args:
            min_df: Minimum document frequency for terms to be included in vocabulary
                (int for absolute count, float for document fraction).
            max_df: Maximum document frequency for terms to be included
                (int for absolute count, float for document fraction).
            ngram_range: A 2-element list [min_n, max_n] specifying range of n-grams.

        Returns:
            A summary dictionary containing DTM statistics, vocabulary size,
            and matrix sparsity percentage.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        if "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe does not contain a 'text' column.")

        # Parameter validation: ngram_range
        if not isinstance(ngram_range, list) or len(ngram_range) != 2:
            raise ValueError("ngram_range must be a list of exactly two integers [min_n, max_n].")
        if not (isinstance(ngram_range[0], int) and isinstance(ngram_range[1], int)):
            raise ValueError("Elements of ngram_range must be integers.")
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError("ngram_range elements must be positive and in ascending order (min_n <= max_n).")

        # Parameter validation: min_df
        if isinstance(min_df, int) and min_df < 1:
            raise ValueError("Integer min_df must be >= 1.")
        elif isinstance(min_df, float) and not (0.0 <= min_df <= 1.0):
            raise ValueError("Float min_df must be between 0.0 and 1.0.")

        # Parameter validation: max_df
        if isinstance(max_df, int) and max_df < 1:
            raise ValueError("Integer max_df must be >= 1.")
        elif isinstance(max_df, float) and not (0.0 < max_df <= 1.0):
            raise ValueError("Float max_df must be between 0.0 and 1.0.")

        # Standardize text column to string and convert nulls to empty string
        # This keeps empty/whitespace documents in place so row alignment is preserved.
        texts = session.dataframe["text"].fillna("").astype(str)

        # Fit CountVectorizer on raw text
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=tuple(ngram_range),
        )
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Save feature matrix, names, and fitted vectorizer in session
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        # Calculate sparsity
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            "build_dtm_tool",
            {"min_df": min_df, "max_df": max_df, "ngram_range": ngram_range},
            summary
        )
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates total term frequencies across all documents in the document-term matrix.

        Returns:
            A dictionary with a summary of term frequencies across the whole corpus,
            or an error message if the DTM has not been built yet.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-term matrix (DTM) has not been built yet. Please run build_dtm_tool first."
            }

        # Sum occurrences across all rows (documents) for each column (term)
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_frequencies = {
            term: int(count)
            for term, count in zip(session.feature_names, counts)
        }

        # Sort terms descending by total frequency
        sorted_term_frequencies = dict(
            sorted(term_frequencies.items(), key=lambda item: item[1], reverse=True)
        )

        summary = {
            "result_id": session.next_result_id("tf"),
            "status": "success",
            "vocabulary_size": len(session.feature_names),
            "term_frequencies": sorted_term_frequencies,
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap of a positional slice of the document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include from the start of the vocabulary.
            n_documents: Number of documents (rows) to include from the start of the corpus.

        Returns:
            A dictionary containing the sliced matrix, document/term labels, and status,
            or an error message if the DTM has not been built yet.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-term matrix (DTM) has not been built yet. Please run build_dtm_tool first."
            }

        # Positional slicing of DTM rows and columns
        sliced_matrix = session.feature_matrix[:n_documents, :n_terms].toarray()
        terms_included = session.feature_names[:n_terms]
        doc_indices = list(range(sliced_matrix.shape[0]))

        # Generate heatmap plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms_included,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap (First {len(doc_indices)} Docs, First {len(terms_included)} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure in session state
        session.pending_figure = fig

        matrix_list = sliced_matrix.tolist()

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "status": "success",
            "n_documents": sliced_matrix.shape[0],
            "n_terms": sliced_matrix.shape[1],
            "terms": terms_included,
            "document_ids": doc_indices,
            "matrix": matrix_list,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]

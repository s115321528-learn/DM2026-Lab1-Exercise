"""Data processing tools for the agentic data mining pipeline."""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
import nltk


def _ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


def make_tools(session: Any) -> list:
    """Factory function to build data tools bound to a session."""

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
        """Tokenize text column in session.dataframe into unigram tokens using NLTK word_tokenize.

        Args:
            text_column: Name of the text column in session.dataframe to tokenize.

        Returns:
            Dict containing status, result_id, and summary metrics.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session.dataframe"}

        if text_column not in session.dataframe.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found in dataframe."}

        _ensure_nltk_resources()

        tokens_list = []
        for val in session.dataframe[text_column]:
            if val is None or not isinstance(val, str) or not val.strip():
                tokens_list.append([])
            else:
                tokens = nltk.word_tokenize(val.lower())
                tokens_list.append(tokens)

        session.dataframe["unigrams"] = tokens_list
        session.artifacts["unigrams"] = tokens_list

        doc_count = len(tokens_list)
        total_tokens = sum(len(t) for t in tokens_list)
        avg_tokens = (total_tokens / doc_count) if doc_count > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "status": "success",
            "n_documents": doc_count,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 4)
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """List files and subdirectories within the specified directory.

        Args:
            subdirectory: Path relative to workspace root (default ".").

        Returns:
            Dict containing status, result_id, sorted directories, and sorted files.
        """
        base_dir = Path(".").resolve()
        target_dir = (base_dir / subdirectory).resolve()

        try:
            target_dir.relative_to(base_dir)
        except ValueError:
            return {"status": "error", "message": "Path outside workspace boundary is rejected."}

        if not target_dir.exists() or not target_dir.is_dir():
            return {
                "result_id": session.next_result_id("list_files"),
                "status": "error",
                "message": f"Directory '{subdirectory}' does not exist.",
                "directories": [],
                "files": []
            }

        files = []
        directories = []

        for entry in target_dir.iterdir():
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        files.sort()
        directories.sort()

        summary = {
            "result_id": session.next_result_id("list_files"),
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """Inspect the working session.dataframe by taking a preview of the first few rows.

        Args:
            n_rows: Number of rows to inspect (default 5, capped at 20).

        Returns:
            Dict containing dataset summary and sample rows.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
            return {"status": "error", "message": "n_rows must be a positive integer."}

        actual_rows = min(n_rows, 20)
        preview_df = session.dataframe.head(actual_rows)
        raw_rows = preview_df.to_dict(orient="records")

        # Ensure JSON-serializable preview rows (parse NaN constant cleanly to None)
        json_safe_str = json.dumps(raw_rows, default=str)
        preview_rows = json.loads(json_safe_str, parse_constant=lambda _: None)

        summary = {
            "result_id": session.next_result_id("inspect"),
            "status": "success",
            "total_rows": len(session.dataframe),
            "columns": list(session.dataframe.columns),
            "preview_rows": preview_rows
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """Check for missing or empty text values in session.dataframe.

        Args:
            text_column: Name of the text column to inspect for missing/empty values (default "text").

        Returns:
            Dict containing result_id, status, missing_count, and total_rows.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found in dataframe."}

        series = session.dataframe[text_column]
        is_na_mask = series.isna()
        is_empty_str_mask = series.apply(lambda val: isinstance(val, str) and not val.strip())

        missing_count = int((is_na_mask | is_empty_str_mask).sum())

        summary = {
            "result_id": session.next_result_id("check_missing"),
            "status": "success",
            "missing_count": missing_count,
            "total_rows": len(session.dataframe)
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(subset_column: str = "text", drop: bool = False) -> Dict[str, Any]:
        """Check for duplicate documents in session.dataframe and optionally drop them using keep=False.

        Args:
            subset_column: Column name to check duplicates on (default "text").
            drop: If True, drop all copies of duplicated rows using keep=False in place.

        Returns:
            Dict containing result_id, status, duplicate_count, n_documents_before, and n_documents_after.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if subset_column not in session.dataframe.columns:
            return {"status": "error", "message": f"Column '{subset_column}' not found in dataframe."}

        n_before = len(session.dataframe)
        
        # Calculate duplicates using default keep='first' logic for counting duplicate rows beyond the first copy
        dup_mask = session.dataframe.duplicated(subset=[subset_column], keep="first")
        duplicate_count = int(dup_mask.sum())

        if drop:
            # Use keep=False (drops every copy of a duplicated row, matching Master's behavior)
            session.dataframe.drop_duplicates(subset=[subset_column], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        n_after = len(session.dataframe)

        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "status": "success",
            "duplicate_count": duplicate_count,
            "n_documents_before": n_before,
            "n_documents_after": n_after,
            "dropped": drop
        }

        session.store_result("check_duplicates_tool", {"subset_column": subset_column, "drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Subsamples the working session.dataframe using pandas sample.

        Args:
            n: Number of rows to sample. Cannot be combined with frac.
            frac: Fraction of axis items to return (0.0 < frac <= 1.0). Cannot be combined with n.
            random_state: Seed for the random number generator (default 42).

        Returns:
            Dict containing result_id, status, sampled_indices, n_samples, and original_total.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        original_total = len(session.dataframe)

        if n is not None and frac is not None:
            return {"status": "error", "message": "Cannot specify both n and frac."}

        if n is None and frac is None:
            n = 5  # Standard default if neither is provided

        if n is not None:
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                return {"status": "error", "message": "n must be a positive integer."}
            if n > original_total:
                return {"status": "error", "message": f"n ({n}) cannot exceed total rows ({original_total})."}

        if frac is not None:
            if not isinstance(frac, (int, float)) or isinstance(frac, bool) or frac <= 0.0 or frac > 1.0:
                return {"status": "error", "message": "frac must be a float/int in (0.0, 1.0]."}

        try:
            sampled_df = session.dataframe.sample(n=n, frac=frac, random_state=random_state)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()]
        session.dataframe = sampled_df.reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("sample_data"),
            "status": "success",
            "sampled_indices": sampled_indices,
            "n_samples": len(sampled_df),
            "original_total": original_total
        }

        session.store_result("sample_data_tool", {"n": n, "frac": frac, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool(text_column: str = "text") -> Dict[str, Any]:
        """Compute text length summary statistics and set a category box plot on session.pending_figure.

        Args:
            text_column: Text column to compute character length statistics for (default "text").

        Returns:
            Dict containing result_id, status, overall stats, and per-category stats.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found in dataframe."}

        # Identify category column name
        cat_col = None
        if "category_name" in session.dataframe.columns:
            cat_col = "category_name"
        elif "category" in session.dataframe.columns:
            cat_col = "category"

        # Calculate character length without modifying session.dataframe
        text_lengths = session.dataframe[text_column].fillna("").astype(str).apply(len)

        # Helper to convert pandas .describe() output dict values to plain Python floats or None if non-finite
        def _format_stats(series_desc: Any) -> Dict[str, Any]:
            desc_dict = series_desc.to_dict()
            formatted = {}
            for k, v in desc_dict.items():
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    formatted[str(k)] = None
                else:
                    formatted[str(k)] = round(float(v), 4)
            return formatted

        overall_stats = _format_stats(text_lengths.describe())

        # Compute per-category stats and generate boxplot if category column is present
        by_category_stats: Dict[str, Dict[str, Any]] = {}
        if cat_col is not None:
            plot_df = session.dataframe.copy()
            plot_df["text_length"] = text_lengths

            for cat_val, group in plot_df.groupby(cat_col):
                by_category_stats[str(cat_val)] = _format_stats(group["text_length"].describe())

            # Build boxplot of text_length grouped by category_name
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=plot_df, x=cat_col, y="text_length", ax=ax)
            ax.set_title(f"Text Length Distribution by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel("Text Length (characters)")
            fig.tight_layout()

            session.pending_figure = fig
        else:
            session.pending_figure = None

        summary = {
            "result_id": session.next_result_id("describe_data"),
            "status": "success",
            "overall": overall_stats,
            "by_category": by_category_stats
        }

        session.store_result("describe_data_tool", {"text_column": text_column}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]

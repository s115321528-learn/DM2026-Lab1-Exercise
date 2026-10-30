"""tools_patterns.py

Frequent pattern mining tools using PAMI algorithms.
"""

from typing import Literal, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    """Factory function creating pattern mining tools bound to the session state."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a specific category using PAMI algorithms.

        Args:
            category_name: Target category to mine patterns for (filters session.dataframe).
            filtering_method: Feature selection method ('variance', 'tfidf', or 'term_frequency'). Default is 'tfidf'.
            algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Default is 'fpgrowth'.
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of top patterns to return (required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, patterns list, and note.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No DataFrame found in session. Run load_dataset_tool first."}

        df = session.dataframe

        # Identify category column
        cat_col = None
        if "category_name" in df.columns:
            cat_col = "category_name"
        elif "category" in df.columns:
            cat_col = "category"
        else:
            return {"error": "DataFrame missing category/category_name column."}

        # Filter documents for category_name
        cat_df = df[df[cat_col].astype(str) == str(category_name)]
        if cat_df.empty:
            return {"error": f"Category '{category_name}' not found in dataset."}

        # Validate algorithm-specific parameters
        if algorithm in ("fpgrowth", "maxfpgrowth"):
            if min_sup is None or min_sup <= 0:
                return {"error": f"min_sup must be a positive integer for algorithm '{algorithm}'."}
        elif algorithm == "topk":
            if k is None or k <= 0:
                return {"error": "k must be a positive integer for algorithm 'topk'."}

        art_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        # Helper for storing empty zero-term result in both session.artifacts and session.store_result
        def _store_empty_result(note_msg: str) -> Dict[str, Any]:
            session.artifacts[art_key] = []
            res_id = session.next_result_id(f"mine_patterns_{category_name}")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": note_msg,
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "algorithm": algorithm}, summary)
            return summary

        # 1. Build category-local DTM with English stopwords dropped
        texts = cat_df["text"].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError:
            X_cat = None

        if X_cat is None or X_cat.shape[1] == 0:
            return _store_empty_result("no terms survived filtering")

        feature_names = np.array(vectorizer.get_feature_names_out())
        X_arr = X_cat.toarray()

        # 2. Apply filtering method
        if filtering_method in ("variance", "term_frequency"):
            if filtering_method == "variance":
                scores = np.var(X_arr, axis=0)
            else:
                scores = np.sum(X_arr, axis=0)

            # Keep middle 90% (strictly between 5th and 95th percentiles)
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            keep_mask = (scores > p5) & (scores < p95)

        elif filtering_method == "tfidf":
            tfidf_trans = TfidfTransformer()
            X_tfidf = tfidf_trans.fit_transform(X_cat).toarray()
            scores = np.mean(X_tfidf, axis=0)

            # Keep top 80% (>= 20th percentile)
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20
        else:
            return {"error": f"Unsupported filtering_method: {filtering_method}"}

        survived_features = feature_names[keep_mask]
        X_survived = X_arr[:, keep_mask]

        # 3. Check zero terms kept
        if len(survived_features) == 0 or X_survived.shape[1] == 0:
            return _store_empty_result("no terms survived filtering")

        # 4. Build transactional DB DataFrame for PAMI DF2DB
        tx_df = pd.DataFrame(X_survived, columns=survived_features)

        import tempfile
        import os
        import PAMI.extras.convert.DF2DB as db_converter

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp_db:
            db_file_path = tmp_db.name

        try:
            conv = db_converter.DF2DB(tx_df, DFtype="dense")
            conv.convert2TransactionalDatabase(
                oFile=db_file_path,
                condition=">=",
                thresholdValue=1
            )

            # Mine with requested PAMI algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(iFile=db_file_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(iFile=db_file_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(iFile=db_file_path, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()
        finally:
            if os.path.exists(db_file_path):
                os.remove(db_file_path)

        # Process mined patterns
        patterns_list = []
        if patterns_df is not None and not patterns_df.empty:
            pat_col = next((c for c in patterns_df.columns if "pattern" in c.lower()), patterns_df.columns[0])
            sup_col = next((c for c in patterns_df.columns if "support" in c.lower()), patterns_df.columns[1])

            for _, row in patterns_df.iterrows():
                pat_str = str(row[pat_col]).strip()
                pat_items = [item.strip() for item in pat_str.split() if item.strip()]
                sup_val = int(row[sup_col])
                patterns_list.append({
                    "pattern": pat_items,
                    "support": sup_val
                })

        session.artifacts[art_key] = patterns_list

        res_id = session.next_result_id(f"mine_patterns_{category_name}")
        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns_list,
            "n_patterns": len(patterns_list),
        }
        session.store_result("mine_patterns_tool", {"category_name": category_name, "algorithm": algorithm}, summary)
        return summary

    return [mine_patterns_tool]

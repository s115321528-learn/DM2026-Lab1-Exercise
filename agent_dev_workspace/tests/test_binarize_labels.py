"""tests/test_binarize_labels.py

Independent tests for binarize_labels_tool in tools_reduction.py.
"""

import pytest
import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools as make_reduction_tools


def _get_binarize_tool(session):
    tools = make_reduction_tools(session)
    return next(t for t in tools if t.name == "binarize_labels_tool")


def test_binarize_labels_normal_multiclass():
    """Test normal one-hot encoding on fixture dataset deriving expected values dynamically."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" not in res
    assert res["n_samples"] == 8
    
    cats = res["categories"]
    matrix = res["binarized_labels"]
    assert len(matrix) == 8

    # Derive expected dynamically from session.dataframe["category_name"] or "label"
    labels_series = session.dataframe["category_name"] if "category_name" in session.dataframe.columns else session.dataframe["label"]
    for idx, raw_label in enumerate(labels_series):
        cat_index = cats.index(str(raw_label))
        expected_row = [1 if i == cat_index else 0 for i in range(len(cats))]
        assert matrix[idx] == expected_row


def test_binarize_labels_fallback_label_column():
    """Test binarize_labels_tool when session.labels is None but session.dataframe['label'] exists."""
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["doc1", "doc2", "doc3"],
        "label": ["catA", "catB", "catA"]
    })

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" not in res
    assert res["categories"] == ["catA", "catB"]
    assert res["binarized_labels"] == [[1, 0], [0, 1], [1, 0]]


def test_binarize_labels_binary_classification():
    """Test binary classification produces exactly two binary columns."""
    session = SessionState()
    session.labels = ["catA", "catB", "catA", "catB"]

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" not in res
    assert res["categories"] == ["catA", "catB"]
    matrix = res["binarized_labels"]
    assert matrix == [[1, 0], [0, 1], [1, 0], [0, 1]]


def test_binarize_labels_single_category():
    """Test single category produces a 1-column matrix."""
    session = SessionState()
    session.labels = ["catA", "catA", "catA"]

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" not in res
    assert res["categories"] == ["catA"]
    assert res["binarized_labels"] == [[1], [1], [1]]


def test_binarize_labels_missing_null_nan():
    """Test error handling when category labels contain NaN/None/empty strings."""
    session = SessionState()
    session.labels = ["catA", None, "catB"]

    binarize_tool = _get_binarize_tool(session)
    res_none = binarize_tool.invoke({})
    assert "error" in res_none

    session_nan = SessionState()
    session_nan.labels = ["catA", np.nan, "catB"]
    binarize_tool_nan = _get_binarize_tool(session_nan)
    res_nan = binarize_tool_nan.invoke({})
    assert "error" in res_nan


def test_binarize_labels_mismatch_count():
    """Test error when session.labels length differs from session.dataframe row count."""
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["a", "b", "c"], "category_name": ["catA", "catA", "catB"]})
    session.labels = ["catA", "catA"]  # 2 labels vs 3 rows

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" in res
    assert "misalignment" in res["error"].lower()


def test_binarize_labels_missing_all_labels():
    """Test error when no labels or category columns exist in session."""
    session = SessionState()
    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" in res

    session_no_cat = SessionState()
    session_no_cat.dataframe = pd.DataFrame({"text": ["a", "b"]})
    binarize_tool_no_cat = _get_binarize_tool(session_no_cat)
    res_no_cat = binarize_tool_no_cat.invoke({})
    assert "error" in res_no_cat


def test_binarize_labels_preserves_data_and_order():
    """Test original document order and non-mutation of session data."""
    session = SessionState()
    orig_labels = ["catB", "catA", "catB"]
    session.labels = orig_labels.copy()
    session.dataframe = pd.DataFrame({"text": ["t1", "t2", "t3"], "category_name": orig_labels})

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" not in res
    assert session.labels == orig_labels
    assert session.dataframe["category_name"].tolist() == orig_labels

    matrix = res["binarized_labels"]
    assert matrix[0] == [0, 1]
    assert matrix[1] == [1, 0]
    assert matrix[2] == [0, 1]


def test_binarize_labels_persistence():
    """Verify state persistence in session.artifacts and session.results_store."""
    session = SessionState()
    session.labels = ["catA", "catB"]

    binarize_tool = _get_binarize_tool(session)
    res = binarize_tool.invoke({})

    assert "error" not in res
    assert "binarized_labels" in session.artifacts
    assert session.artifacts["binarized_labels"] == res["binarized_labels"]
    assert session.artifacts["binarized_label_classes"] == res["categories"]

    res_id = res["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].summary["binarized_labels"] == res["binarized_labels"]

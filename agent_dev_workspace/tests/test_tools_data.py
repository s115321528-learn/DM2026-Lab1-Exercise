"""Tests for tools_data.py tokenization, file listing, data inspection, check_missing_tool, check_duplicates_tool, sample_data_tool, and describe_data_tool functionality."""

import json
import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_known_answer_10():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = tokenize_tool.invoke({"text_column": "text"})
    assert res["status"] == "success"
    assert res["n_documents"] == 8

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens


def test_tokenize_tool_missing_column():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["hello world"]})
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"text_column": "non_existent"})
    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_tokenize_tool_empty_and_nan_text():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["hello world", "", None, "  "]})
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"text_column": "text"})
    assert res["status"] == "success"
    assert session.dataframe["unigrams"].tolist() == [
        ["hello", "world"],
        [],
        [],
        []
    ]


def test_tokenize_tool_state_persistence():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["sample doc"]})
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"text_column": "text"})
    result_id = res["result_id"]

    assert "unigrams" in session.dataframe.columns
    assert session.artifacts["unigrams"] == [["sample", "doc"]]
    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].tool_name == "tokenize_tool"


def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["status"] == "success"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_missing_directory():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "non_existent_folder_123"})
    assert res["status"] == "error"
    assert res["directories"] == []
    assert res["files"] == []


def test_list_files_tool_outside_boundary():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "../../.."})
    assert res["status"] == "error"
    assert "outside workspace boundary" in res["message"]


def test_inspect_data_tool_basic_and_capping():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": [f"doc_{i}" for i in range(30)]})
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 25})
    assert res["status"] == "success"
    assert res["total_rows"] == 30
    assert len(res["preview_rows"]) == 20


def test_inspect_data_tool_nan_and_literal_text_regression():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["NaN technology", None, np.nan],
        "val": [1.0, np.nan, 3.0]
    })
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 5})
    assert res["status"] == "success"
    rows = res["preview_rows"]
    
    # Verify "NaN technology" literal is preserved intact
    assert rows[0]["text"] == "NaN technology"
    # Verify np.nan / None are converted to None (JSON null)
    assert rows[1]["text"] is None
    assert rows[1]["val"] is None


def test_inspect_data_tool_invalid_n_rows():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["a", "b"]})
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res_zero = inspect_data_tool.invoke({"n_rows": 0})
    assert res_zero["status"] == "error"

    res_neg = inspect_data_tool.invoke({"n_rows": -5})
    assert res_neg["status"] == "error"


def test_inspect_data_tool_state_persistence():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["doc1"]})
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 1})
    result_id = res["result_id"]

    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].tool_name == "inspect_data_tool"


def test_check_missing_tool_known_answer_2():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = check_missing_tool.invoke({"text_column": "text"})
    assert res["status"] == "success"
    assert res["missing_count"] == 1
    assert res["total_rows"] == 8


def test_check_missing_tool_various_missing_types():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["valid", None, np.nan, pd.NA, "", "   "]
    })
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"text_column": "text"})
    assert res["status"] == "success"
    assert res["missing_count"] == 5
    assert res["total_rows"] == 6


def test_check_missing_tool_valid_numerics():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "num_col": [0, 0.0, 123.45, -1.0, np.nan]
    })
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"text_column": "num_col"})
    assert res["status"] == "success"
    assert res["missing_count"] == 1
    assert res["total_rows"] == 5


def test_check_missing_tool_error_handling():
    session = SessionState()
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Missing dataframe
    res_no_df = check_missing_tool.invoke({"text_column": "text"})
    assert res_no_df["status"] == "error"

    # Missing column
    session.dataframe = pd.DataFrame({"colA": ["data"]})
    res_no_col = check_missing_tool.invoke({"text_column": "non_existent"})
    assert res_no_col["status"] == "error"


def test_check_missing_tool_state_persistence():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["doc1", None]})
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"text_column": "text"})
    result_id = res["result_id"]

    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].tool_name == "check_missing_tool"


def test_check_duplicates_tool_known_answer_3():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test drop=False: duplicate count is 1, dataframe remains length 8
    res_check = check_duplicates_tool.invoke({"subset_column": "text", "drop": False})
    assert res_check["status"] == "success"
    assert res_check["duplicate_count"] == 1
    assert res_check["n_documents_before"] == 8
    assert res_check["n_documents_after"] == 8
    assert len(session.dataframe) == 8

    # Test drop=True: keep=False drops both copies of duplicate row, leaving exactly 6 rows
    res_drop = check_duplicates_tool.invoke({"subset_column": "text", "drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["duplicate_count"] == 1
    assert res_drop["n_documents_before"] == 8
    assert res_drop["n_documents_after"] == 6
    assert len(session.dataframe) == 6


def test_check_duplicates_tool_error_handling():
    session = SessionState()
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Missing dataframe
    res_no_df = check_duplicates_tool.invoke({"subset_column": "text"})
    assert res_no_df["status"] == "error"

    # Missing column
    session.dataframe = pd.DataFrame({"colA": ["doc1"]})
    res_no_col = check_duplicates_tool.invoke({"subset_column": "non_existent"})
    assert res_no_col["status"] == "error"


def test_check_duplicates_tool_state_persistence():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["doc1", "doc1"]})
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"subset_column": "text"})
    result_id = res["result_id"]

    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].tool_name == "check_duplicates_tool"


def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["status"] == "success"
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["n_samples"] == 4
    assert res["original_total"] == 8
    assert len(session.dataframe) == 4


def test_sample_data_tool_validation_and_errors():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": [f"doc_{i}" for i in range(10)]})
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Reject both n and frac
    res_both = sample_data_tool.invoke({"n": 3, "frac": 0.5})
    assert res_both["status"] == "error"

    # Reject n > total rows
    res_exceed = sample_data_tool.invoke({"n": 15})
    assert res_exceed["status"] == "error"


def test_sample_data_tool_state_persistence():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["a", "b", "c"]})
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 2, "random_state": 42})
    result_id = res["result_id"]

    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].tool_name == "sample_data_tool"


def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    initial_df = session.dataframe.copy()

    res = describe_data_tool.invoke({"text_column": "text"})
    assert res["status"] == "success"

    # Overall stats
    overall = res["overall"]
    assert overall["count"] == 8.0
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0.0
    assert overall["25%"] == 17.75
    assert overall["50%"] == 20.5
    assert overall["75%"] == 25.25
    assert overall["max"] == 30.0

    # Per category stats
    by_cat = res["by_category"]
    assert "catA" in by_cat and "catB" in by_cat
    
    catA = by_cat["catA"]
    assert catA["count"] == 4.0
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert catA["max"] == 29.0

    catB = by_cat["catB"]
    assert catB["count"] == 4.0
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["max"] == 30.0

    # Check figure lifecycle
    assert session.pending_figure is not None

    # Check non-mutation of dataframe
    pd.testing.assert_frame_equal(session.dataframe, initial_df)

    # Check strict JSON serializability (allow_nan=False)
    json_str = json.dumps(res, allow_nan=False)
    assert isinstance(json_str, str)


def test_describe_data_tool_single_row_category_and_no_cat_col():
    session = SessionState()
    # One category has 1 row (producing std=NaN from pandas describe)
    session.dataframe = pd.DataFrame({
        "text": ["hello", "world doc"],
        "category_name": ["cat1", "cat2"]
    })
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text"})
    assert res["status"] == "success"
    # std for cat1 with 1 row should be formatted as None (JSON null)
    assert res["by_category"]["cat1"]["std"] is None
    assert session.pending_figure is not None

    # Test without category column: pending_figure should be set to None
    session_no_cat = SessionState()
    session_no_cat.dataframe = pd.DataFrame({"text": ["doc1", "doc2"]})
    tools_no_cat = make_tools(session_no_cat)
    describe_no_cat = next(t for t in tools_no_cat if t.name == "describe_data_tool")

    res_no_cat = describe_no_cat.invoke({"text_column": "text"})
    assert res_no_cat["status"] == "success"
    assert res_no_cat["by_category"] == {}
    assert session_no_cat.pending_figure is None

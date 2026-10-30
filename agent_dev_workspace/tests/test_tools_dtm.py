import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    session = SessionState()

    # Load load_dataset_tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate build_dtm_tool from tools_dtm
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with defaults
    res = build_dtm_tool.invoke({})

    # Check against Known Answer #4 in TEST_FIXTURE.md
    # Vocabulary: alpha, always, beta, delta, gamma (5 terms)
    assert res["vocabulary_size"] == 5
    assert sorted(session.feature_names) == ["alpha", "always", "beta", "delta", "gamma"]

    # Sparsity: 21 non-zero cells out of 40 total (8 rows * 5 columns) -> sparsity_pct = 47.5
    assert res["n_documents"] == 8
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5, atol=1e-3)

    # Check session state and artifacts populated
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    tools = make_tools(session)
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # 1. Verify handling of uninitialized DTM
    res_error = term_frequency_tool.invoke({})
    assert res_error["status"] == "error"
    assert "not been built yet" in res_error["message"]

    # 2. Setup DTM via load_dataset_tool and build_dtm_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Invoke term_frequency_tool
    res = term_frequency_tool.invoke({})

    # 4. Check status and stored result
    assert res["status"] == "success"
    assert "result_id" in res
    result_id = res["result_id"]
    assert result_id in session.results_store

    # 5. Check against Known Answer #14 in TEST_FIXTURE.md
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_freqs = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }
    assert res["vocabulary_size"] == 5
    assert res["term_frequencies"] == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    tools = make_tools(session)
    build_dtm = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # 1. Verify handling of uninitialized DTM
    res_error = dtm_heatmap.invoke({})
    assert res_error["status"] == "error"
    assert "not been built yet" in res_error["message"]

    build_dtm.invoke({})

    # 2. Full matrix call (defaults: n_terms=20, n_documents=20)
    full_res = dtm_heatmap.invoke({"n_terms": 20, "n_documents": 20})

    assert full_res["status"] == "success"
    assert "result_id" in full_res
    assert full_res["result_id"] in session.results_store
    assert full_res["n_documents"] == 8
    assert full_res["n_terms"] == 5
    assert full_res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert full_res["matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # 3. Slice matrix call (n_terms=2, n_documents=3)
    slice_res = dtm_heatmap.invoke({"n_terms": 2, "n_documents": 3})

    assert slice_res["status"] == "success"
    assert "result_id" in slice_res
    assert slice_res["result_id"] in session.results_store
    assert slice_res["n_documents"] == 3
    assert slice_res["n_terms"] == 2
    assert slice_res["terms"] == ["alpha", "always"]
    assert slice_res["matrix"] == [[3, 1], [2, 1], [1, 1]]

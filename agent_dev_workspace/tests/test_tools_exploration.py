import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools


def test_feature_correlation_matrix_tool_known_answer_19():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_n": 20})

    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["selected_terms"] == expected_terms

    expected_matrix = np.array([
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])

    actual_matrix = np.array(res["correlation_matrix"])
    assert np.allclose(actual_matrix, expected_matrix, atol=1e-3)
    assert session.pending_figure is not None


def test_feature_correlation_matrix_tool_error_handling():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    assert "error" in feature_corr_tool.invoke({"top_n": 0})
    assert "error" in feature_corr_tool.invoke({"top_n": -5})


def test_cosine_similarity_tool_known_answer_16():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16 pairs
    res_0_1 = cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_3_4 = cosine_tool.invoke({"doc_idx_1": 3, "doc_idx_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_7 = cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 7})
    assert res_0_7["similarity"] == 0.0


def test_cosine_similarity_tool_error_handling():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    assert "error" in cosine_tool.invoke({"doc_idx_1": -1, "doc_idx_2": 0})
    assert "error" in cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 10})

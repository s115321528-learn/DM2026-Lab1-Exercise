import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools


def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load build_dtm_tool to populate session.feature_matrix and session.feature_names
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load variance_filter_tool
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Test threshold=0.15 against Known Answer #5
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]

    # Verify variances stored in full_report match Known Answer #5
    report = session.results_store[res["result_id"]]["full_report"]
    term_var_map = dict(zip(report["term"], report["variance"]))

    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    for term, exp_var in expected_variances.items():
        assert np.isclose(term_var_map[term], exp_var, atol=1e-3)


def test_variance_filter_tool_invalid_threshold():
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

    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": -0.5})
    assert "error" in res


def test_pearson_filter_tool_known_answer_6():
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

    filtering_tools = make_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Test target_class="catB" with threshold=0.0 against Known Answer #6
    res = pearson_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert res["target_class"] == "catB"

    report = session.results_store[res["result_id"]]["full_report"]
    term_r_map = dict(zip(report["term"], report["pearson_r"]))

    expected_corrs = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }
    for term, exp_r in expected_corrs.items():
        assert term_r_map[term] == pytest.approx(exp_r, abs=1e-3)

    # Test threshold=0.7: abs(r) >= 0.7 keeps delta, gamma, beta; removes alpha, always
    res_thresh = pearson_tool.invoke({"target_class": "catB", "threshold": 0.7})
    assert res_thresh["n_kept"] == 3
    assert set(res_thresh["kept_terms"]) == {"delta", "gamma", "beta"}
    assert set(res_thresh["removed_terms"]) == {"alpha", "always"}


def test_pearson_filter_tool_error_handling():
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

    filtering_tools = make_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    assert "error" in pearson_tool.invoke({"target_class": "catB", "threshold": -0.1})
    assert "error" in pearson_tool.invoke({"target_class": "catB", "threshold": 1.5})
    assert "error" in pearson_tool.invoke({"target_class": "nonexistent_class", "threshold": 0.0})


def test_spearman_filter_tool_known_answer_6():
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

    filtering_tools = make_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Test target_class="catB" with threshold=0.0 against Known Answer #6
    res = spearman_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert res["target_class"] == "catB"

    report = session.results_store[res["result_id"]]["full_report"]
    term_r_map = dict(zip(report["term"], report["spearman_r"]))

    expected_corrs = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }
    for term, exp_r in expected_corrs.items():
        assert term_r_map[term] == pytest.approx(exp_r, abs=1e-3)

    # Test threshold=0.8: abs(r) >= 0.8 keeps delta, gamma; removes alpha, beta, always
    res_thresh = spearman_tool.invoke({"target_class": "catB", "threshold": 0.8})
    assert res_thresh["n_kept"] == 2
    assert set(res_thresh["kept_terms"]) == {"delta", "gamma"}
    assert set(res_thresh["removed_terms"]) == {"alpha", "beta", "always"}


def test_spearman_filter_tool_error_handling():
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

    filtering_tools = make_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    assert "error" in spearman_tool.invoke({"target_class": "catB", "threshold": -0.1})
    assert "error" in spearman_tool.invoke({"target_class": "catB", "threshold": 1.5})
    assert "error" in spearman_tool.invoke({"target_class": "nonexistent_class", "threshold": 0.0})

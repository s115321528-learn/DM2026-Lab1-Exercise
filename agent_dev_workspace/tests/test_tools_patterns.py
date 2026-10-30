"""tests/test_tools_patterns.py

Tests for mine_patterns_tool in tools_patterns.py against Known Answers #17 and #20 in TEST_FIXTURE.md.
"""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    return session


def test_mine_patterns_fpgrowth_known_answer_17():
    """Test catA and catB pattern mining with tfidf + fpgrowth against Known Answer #17."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA + tfidf + fpgrowth (min_sup=2)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 2
    })
    assert "error" not in res_catA
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # catB + tfidf + fpgrowth (min_sup=2)
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 2
    })
    assert "error" not in res_catB
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_zero_surviving_terms():
    """Test variance and term_frequency filtering produce zero surviving terms on sample fixture."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # variance filtering for catA
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_var
    assert res_var["patterns"] == []
    assert res_var["note"] == "no terms survived filtering"

    # term_frequency filtering for catB
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_tf
    assert res_tf["patterns"] == []
    assert res_tf["note"] == "no terms survived filtering"


def test_mine_patterns_topk_known_answer_20():
    """Test algorithm='topk' with k=1, 2, 3 against Known Answer #20."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    for k_val in (1, 2, 3):
        res_catA = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert "error" not in res_catA
        assert len(res_catA["patterns"]) == 1
        assert res_catA["patterns"][0]["pattern"] == ["alpha"]
        assert res_catA["patterns"][0]["support"] == 3

        res_catB = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert "error" not in res_catB
        assert len(res_catB["patterns"]) == 1
        assert res_catB["patterns"][0]["pattern"] == ["gamma"]
        assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_maxfpgrowth_known_answer_20():
    """Test algorithm='maxfpgrowth' returns empty patterns against Known Answer #20."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    for min_sup_val in (1, 4):
        res_catA = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup_val
        })
        assert "error" not in res_catA
        assert res_catA["patterns"] == []

        res_catB = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup_val
        })
        assert "error" not in res_catB
        assert res_catB["patterns"] == []


def test_mine_patterns_invalid_parameters():
    """Test error returns for invalid category name and invalid mining parameters."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Non-existent category
    res = mine_patterns_tool.invoke({
        "category_name": "NonExistentCat",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res

    # Default min_sup=10 should be accepted
    res_default_sup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" not in res_default_sup

    # Invalid min_sup=0 should be rejected
    res_zero_sup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 0
    })
    assert "error" in res_zero_sup

    # Negative min_sup should be rejected
    res_negative_sup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": -1
    })
    assert "error" in res_negative_sup

    # Missing k for topk
    res_no_k = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in res_no_k


def test_mine_patterns_artifacts_and_results_store():
    """Verify mined patterns and empty results are stored in session.artifacts and session.results_store."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Mining run
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 2
    })
    assert "error" not in res

    # Check artifact key
    art_key = "patterns_catA_tfidf_fpgrowth"
    assert art_key in session.artifacts
    assert session.artifacts[art_key] == res["patterns"]

    # Check session.results_store via ResultRecord summary
    res_id = res["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].summary["patterns"] == res["patterns"]

    # Zero-term run artifact check
    res_zero = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    art_key_zero = "patterns_catA_variance_fpgrowth"
    assert art_key_zero in session.artifacts
    assert session.artifacts[art_key_zero] == []
    assert res_zero["result_id"] in session.results_store

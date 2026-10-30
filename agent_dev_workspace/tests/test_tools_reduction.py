"""tests/test_tools_reduction.py

Tests for reduce_dimensions_tool in tools_reduction.py:
1. Known Answer #7 (PCA)
2. Known Answer #8 (t-SNE reference coordinates)
3. Known Answer #9 (UMAP reference coordinates)
4. Functional / version-independent tests for t-SNE and UMAP (shape, finite values, figure creation, category legend, and consistency with direct estimator calls)
5. Error handling for invalid parameters and missing labels
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest
import matplotlib.figure
from sklearn.manifold import TSNE
import umap
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_pca_reduction_known_answer_7():
    """Test PCA reduction against Known Answer #7 in TEST_FIXTURE.md."""
    session = _setup_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert "error" not in res
    assert res["n_samples"] == 8
    assert res["n_components"] == 2

    # Check session.pending_figure and category labels in plot
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    ax = session.pending_figure.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert set(legend_texts) == {"catA", "catB"}

    # Known Answer #7 PCA coordinates
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_var = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    assert np.allclose(coords, expected_coords, atol=1e-3)

    var_ratio = res["explained_variance_ratio"]
    assert np.allclose(var_ratio, expected_var, atol=1e-3)


def test_tsne_reduction_known_answer_8():
    """Test t-SNE reduction against Known Answer #8 in TEST_FIXTURE.md."""
    session = _setup_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })

    assert "error" not in res, f"Expected no error, got {res}"
    assert res["n_samples"] == 8
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ])

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    diff = np.abs(coords - expected_coords)
    assert np.allclose(coords, expected_coords, atol=1e-1), f"TSNE mismatch max diff: {diff.max()}\nCoords:\n{coords}"


def test_umap_reduction_known_answer_9():
    """Test UMAP reduction against Known Answer #9 in TEST_FIXTURE.md."""
    session = _setup_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    assert "error" not in res, f"Expected no error, got {res}"
    assert res["n_samples"] == 8
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    diff = np.abs(coords - expected_coords)
    assert np.allclose(coords, expected_coords, atol=1e-1), f"UMAP mismatch max diff: {diff.max()}\nCoords:\n{coords}"


def test_tsne_functional():
    """Version-independent functional test for t-SNE reduction."""
    session = _setup_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    assert "error" not in res
    assert res["n_samples"] == 8
    assert res["n_components"] == 2

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    assert np.all(np.isfinite(coords))

    # Check pending figure & legend
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    ax = session.pending_figure.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert set(legend_texts) == {"catA", "catB"}

    # Consistency with direct TSNE call on same matrix and params
    X_dense = session.feature_matrix.toarray()
    direct_coords = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random").fit_transform(X_dense)
    assert np.allclose(coords, direct_coords, atol=1e-5)


def test_umap_functional():
    """Version-independent functional test for UMAP reduction."""
    session = _setup_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert "error" not in res
    assert res["n_samples"] == 8
    assert res["n_components"] == 2

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    assert np.all(np.isfinite(coords))

    # Check pending figure & legend
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    ax = session.pending_figure.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert set(legend_texts) == {"catA", "catB"}

    # Consistency with direct UMAP call on same matrix and params
    X_dense = session.feature_matrix.toarray()
    direct_coords = umap.UMAP(n_components=2, n_neighbors=3, random_state=42).fit_transform(X_dense)
    assert np.allclose(coords, direct_coords, atol=1e-5)


def test_invalid_parameters():
    """Test error handling for invalid parameters and missing labels."""
    session = _setup_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Invalid n_components
    res = reduce_dimensions_tool.invoke({"n_components": 3})
    assert "error" in res

    # Perplexity >= n_samples (8)
    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 10.0})
    assert "error" in res

    # n_neighbors >= n_samples (8)
    res = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 10})
    assert "error" in res

    # Missing labels check
    session_no_labels = SessionState()
    session_no_labels.feature_matrix = session.feature_matrix
    reduction_tools_no_labels = make_reduction_tools(session_no_labels)
    reduce_dim_no_labels = next(t for t in reduction_tools_no_labels if t.name == "reduce_dimensions_tool")

    res_no_labels = reduce_dim_no_labels.invoke({})
    assert "error" in res_no_labels
    assert "labels" in res_no_labels["error"].lower()

"""tests/test_tools_reduction_diag.py

Diagnostic script with single line string in assert False for pytest error output.
"""

import numpy as np
import sklearn
import umap
from sklearn.manifold import TSNE
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


def test_diagnostics():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    X_dense = session.feature_matrix.toarray()

    expected_tsne = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ])

    tsne_default = TSNE(n_components=2, perplexity=2.0, random_state=42).fit_transform(X_dense)
    tsne_random = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random").fit_transform(X_dense)
    tsne_pca = TSNE(n_components=2, perplexity=2.0, random_state=42, init="pca").fit_transform(X_dense)

    err_tsne_default = float(np.max(np.abs(tsne_default - expected_tsne)))
    err_tsne_random = float(np.max(np.abs(tsne_random - expected_tsne)))
    err_tsne_pca = float(np.max(np.abs(tsne_pca - expected_tsne)))

    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])

    umap_coords = umap.UMAP(n_components=2, n_neighbors=3, random_state=42).fit_transform(X_dense)
    err_umap = float(np.abs(umap_coords - expected_umap).max())

    report = (
        f"SKLEARN_VERSION={sklearn.__version__} "
        f"UMAP_VERSION={umap.__version__} "
        f"TSNE_DEF_ERR={err_tsne_default:.4f} "
        f"TSNE_RAND_ERR={err_tsne_random:.4f} "
        f"TSNE_PCA_ERR={err_tsne_pca:.4f} "
        f"UMAP_ERR={err_umap:.4f}"
    )
    assert False, report

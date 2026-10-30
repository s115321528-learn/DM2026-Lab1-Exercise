"""tests/test_pami_diag.py

Diagnostic script to safely inspect PAMI package version, import paths, constructor signatures, and methods.
"""

import importlib
import inspect


def test_pami_signatures():
    pami_info = {}

    try:
        pami_mod = importlib.import_module("PAMI")
        pami_info["PAMI_version"] = getattr(pami_mod, "__version__", "unknown")
    except Exception as e:
        pami_info["PAMI_import_error"] = str(e)

    # Inspect DF2DB
    try:
        df2db_mod = importlib.import_module("PAMI.extras.convert.DF2DB")
        df2db_cls = getattr(df2db_mod, "DF2DB")
        pami_info["DF2DB_path"] = df2db_mod.__file__
        pami_info["DF2DB_init_sig"] = str(inspect.signature(df2db_cls.__init__))
        pami_info["DF2DB_has_convert2TransactionalDatabase"] = hasattr(df2db_cls, "convert2TransactionalDatabase")
    except Exception as e:
        pami_info["DF2DB_error"] = str(e)

    # Inspect FPGrowth
    try:
        fpg_mod = importlib.import_module("PAMI.frequentPattern.basic.FPGrowth")
        fpg_cls = getattr(fpg_mod, "FPGrowth")
        pami_info["FPGrowth_path"] = fpg_mod.__file__
        pami_info["FPGrowth_init_sig"] = str(inspect.signature(fpg_cls.__init__))
    except Exception as e:
        pami_info["FPGrowth_error"] = str(e)

    # Inspect FAE (topk)
    try:
        fae_mod = importlib.import_module("PAMI.frequentPattern.topk.FAE")
        fae_cls = getattr(fae_mod, "FAE")
        pami_info["FAE_path"] = fae_mod.__file__
        pami_info["FAE_init_sig"] = str(inspect.signature(fae_cls.__init__))
    except Exception as e:
        pami_info["FAE_error"] = str(e)

    # Inspect MaxFPGrowth
    try:
        mfpg_mod = importlib.import_module("PAMI.frequentPattern.maximal.MaxFPGrowth")
        mfpg_cls = getattr(mfpg_mod, "MaxFPGrowth")
        pami_info["MaxFPGrowth_path"] = mfpg_mod.__file__
        pami_info["MaxFPGrowth_init_sig"] = str(inspect.signature(mfpg_cls.__init__))
    except Exception as e:
        pami_info["MaxFPGrowth_error"] = str(e)

    report = "\n".join(f"{k}: {v}" for k, v in pami_info.items())
    print("\n--- PAMI DIAGNOSTIC REPORT ---\n" + report)
    assert True, report

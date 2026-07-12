from __future__ import annotations

import importlib


def test_dashboard_module_imports() -> None:
    module = importlib.import_module("app.dashboard")

    assert hasattr(module, "main")


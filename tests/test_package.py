"""Package-level smoke tests."""


def test_package_import() -> None:
    """The src-layout package is importable after installation."""
    import pairs_trading

    assert pairs_trading.__name__ == "pairs_trading"


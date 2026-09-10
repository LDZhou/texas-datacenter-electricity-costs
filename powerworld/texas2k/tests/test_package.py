"""Package-level tests for the Texas2k workflow."""


def test_texas2k_package_imports():
    """The public workflow exposes a stable package version."""
    import powerworld.texas2k

    assert powerworld.texas2k.__version__ == "1.0.0"

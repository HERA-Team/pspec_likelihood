"""Very basic tests of the infrastructure."""


def test_null():
    """Really just a placeholder so that the import is triggered."""
    from pspec_likelihood import DataModelInterface

    assert callable(DataModelInterface)

    assert True

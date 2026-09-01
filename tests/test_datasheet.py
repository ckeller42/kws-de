from pathlib import Path


def test_datasheet_has_required_sections():
    text = Path("docs/DATASHEET.md").read_text(encoding="utf-8")
    for section in (
        "## Motivation",
        "## Composition",
        "## Collection",
        "## Provenance",
        "## Licensing",
        "## Splits",
        "## Recommended uses",
        "## Limitations",
    ):
        assert section in text, f"missing datasheet section: {section}"

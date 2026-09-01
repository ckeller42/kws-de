from kws_de.benchmark import render_table


def test_render_table_has_all_columns_and_rows():
    rows = [
        {
            "name": "ds_cnn",
            "isolated_acc": 0.95,
            "catalog_acc": 0.69,
            "params": 5351,
            "macs": 2069984,
            "int8_bytes": 20216,
            "budget_ok": True,
        },
        {
            "name": "bc_resnet",
            "isolated_acc": 0.96,
            "catalog_acc": 0.70,
            "params": 12000,
            "macs": 900000,
            "int8_bytes": 15000,
            "budget_ok": True,
        },
    ]
    md = render_table(rows)
    for col in ("Architecture", "Isolated", "Catalog", "Params", "MACs", "INT8", "Budget"):
        assert col in md
    assert "ds_cnn" in md and "bc_resnet" in md


def test_render_table_marks_budget_failure():
    rows = [
        {
            "name": "matchboxnet",
            "isolated_acc": 0.9,
            "catalog_acc": 0.6,
            "params": 100000,
            "macs": 5_000_000,
            "int8_bytes": 600_000,
            "budget_ok": False,
        }
    ]
    md = render_table(rows)
    assert "no" in md
    assert "yes" not in md


def test_render_table_empty_rows_still_has_header():
    md = render_table([])
    assert "Architecture" in md
    assert md.count("\n") >= 2  # header + separator


def test_render_table_float_column_optional():
    base = {"catalog_acc": 0.5, "params": 1, "macs": 1, "int8_bytes": 1, "budget_ok": True}
    md = render_table(
        [
            {"name": "a", "isolated_acc": 0.90, "float_acc": 0.95, **base},
            {"name": "b", "isolated_acc": 0.80, **base},
        ]
    )
    assert "| Float |" in md
    assert "| 0.950 |" in md and "| - |" in md

from kws_de import verify


def test_manifest_lists_every_file_sorted_with_hash_and_size(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("bb")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / ".DS_Store").write_text("junk")  # macOS noise, must be skipped
    rows = verify.manifest(tmp_path)
    rels = [r.split("  ", 2)[2] for r in rows]
    assert rels == ["a.txt", "sub/b.txt"]  # sorted, .DS_Store excluded
    # sha256("a") and the byte size are recorded
    sha, size, rel = rows[0].split("  ", 2)
    assert size == "1" and len(sha) == 64


def test_diff_reports_missing_extra_and_changed(tmp_path):
    a = tmp_path / "a.manifest"
    b = tmp_path / "b.manifest"
    a.write_text("h1  1  same.txt\nh2  2  changed.txt\nh3  3  only_a.txt\n")
    b.write_text("h1  1  same.txt\nhX  9  changed.txt\nh4  4  only_b.txt\n")
    only_a, only_b, changed = verify.diff(a, b)
    assert only_a == ["only_a.txt"]
    assert only_b == ["only_b.txt"]
    assert changed == ["changed.txt"]


def test_diff_of_identical_trees_is_empty(tmp_path):
    tree = tmp_path / "tree"
    (tree / "y").mkdir(parents=True)
    (tree / "x.txt").write_text("hello")
    (tree / "y" / "z.bin").write_bytes(b"\x00\x01\x02")
    text = "\n".join(verify.manifest(tree))  # hash once; manifests live outside the tree
    m1, m2 = tmp_path / "m1", tmp_path / "m2"
    m1.write_text(text)
    m2.write_text(text)
    assert verify.diff(m1, m2) == ([], [], [])

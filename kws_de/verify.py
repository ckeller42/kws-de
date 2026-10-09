"""Data-tree integrity: a per-file sha256 manifest, and a diff of two manifests.

The migration cross-check (design §3): build a manifest on each machine, diff
them, and a clean diff is the gate to trust the copy and flip `thinky` to
primary. Pure over the filesystem (no ssh, no rsync) so it is unit-testable;
`scripts/sync-data.sh` does the ssh orchestration around it.

    kws-verify manifest [ROOT] [-o FILE]   # sha256 every file under ROOT
    kws-verify diff A.manifest B.manifest  # report drift; exit 1 if any
"""

import argparse
import hashlib
import sys
from pathlib import Path

from kws_de import config

_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def manifest(root: Path) -> list[str]:
    """`<sha256>  <size>  <relpath>` for every file under `root`, sorted by
    relpath. Symlinks are followed as their target file; `.DS_Store` is skipped
    (macOS noise, never on the Linux side, would show as spurious drift)."""
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == ".DS_Store":
            continue
        rel = p.relative_to(root).as_posix()
        rows.append(f"{_sha256(p)}  {p.stat().st_size}  {rel}")
    return rows


def _read(path: Path) -> dict[str, str]:
    # relpath -> "sha  size"; splitting on the 2-space field sep keeps spaces in paths.
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sha, size, rel = line.split("  ", 2)
        out[rel] = f"{sha}  {size}"
    return out


def diff(a: Path, b: Path) -> tuple[list, list, list]:
    """(only in A, only in B, changed) between two manifest files."""
    ma, mb = _read(a), _read(b)
    only_a = sorted(set(ma) - set(mb))
    only_b = sorted(set(mb) - set(ma))
    changed = sorted(r for r in set(ma) & set(mb) if ma[r] != mb[r])
    return only_a, only_b, changed


def main() -> None:
    ap = argparse.ArgumentParser(prog="kws-verify", description="data-tree sha256 manifest + diff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="write a sha256 manifest of a tree")
    m.add_argument("root", nargs="?", default=None, help="tree to hash (default: the data root)")
    m.add_argument("-o", "--out", default=None, help="write here instead of stdout")
    d = sub.add_parser("diff", help="diff two manifest files; exit 1 on any drift")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()

    if a.cmd == "manifest":
        root = Path(a.root) if a.root else config._DATA_ROOT
        if not root.is_dir():
            print(f"{root}: not a directory (exit 2)", file=sys.stderr)
            raise SystemExit(2)
        rows = manifest(root)
        text = "\n".join(rows) + ("\n" if rows else "")
        if a.out:
            Path(a.out).write_text(text)
            print(f"{len(rows)} files -> {a.out}")
        else:
            sys.stdout.write(text)
        return

    only_a, only_b, changed = diff(Path(a.a), Path(a.b))
    for rel in only_a:
        print(f"only in {a.a}: {rel}")
    for rel in only_b:
        print(f"only in {a.b}: {rel}")
    for rel in changed:
        print(f"changed: {rel}")
    total = len(only_a) + len(only_b) + len(changed)
    if total:
        print(f"DRIFT: {len(only_a)} only-A, {len(only_b)} only-B, {len(changed)} changed")
        raise SystemExit(1)
    print("trees match")


if __name__ == "__main__":
    main()

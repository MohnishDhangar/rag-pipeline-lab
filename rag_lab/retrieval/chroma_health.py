"""Preflight persistent Chroma DBs without embedding API requests."""
from __future__ import annotations
import argparse
from pathlib import Path


def find_indexes(root):
    return sorted({p.parent for p in Path(root).expanduser().resolve().rglob("chroma.sqlite3")})


def inspect_index(path):
    path = Path(path).expanduser().resolve()
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"No Chroma database at {path}. Locate it with "
            "python -m rag_lab.retrieval.chroma_health --search-root ."
        )
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    return {"path": str(path),
            "collections": {name: client.get_collection(name=name).count() for name in names}}


def require_populated_index(path, collection="langchain"):
    report = inspect_index(path)
    count = report["collections"].get(collection, 0)
    if count == 0:
        raise RuntimeError(
            f"Chroma collection {collection!r} is missing or empty at {report['path']}. "
            f"Available collections: {report['collections']}. "
            "Use --chroma-path to select your populated database, possibly under a "
            "literal '~' directory from the foundation notebook. Do not re-embed."
        )
    return count


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search-root", type=Path)
    group.add_argument("--path", type=Path)
    args = parser.parse_args()
    indexes = find_indexes(args.search_root) if args.search_root else [args.path]
    if not indexes: print("No Chroma SQLite databases found.")
    for path in indexes:
        try: print(inspect_index(path))
        except Exception as exc: print(f"{path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

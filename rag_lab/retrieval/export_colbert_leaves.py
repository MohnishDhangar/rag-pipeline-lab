"""Export Chroma passages + metadata in one aligned order before ColBERT indexing.

python -m rag_lab.retrieval.export_colbert_leaves --root .
This replaces the manual notebook dump of leaf_texts.json; rebuild ColBERT
when changing its indexed corpus or metadata mapping.
"""
import argparse, json
from pathlib import Path

def export(root):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
    root=Path(root)
    dense=Chroma(persist_directory=str(root/"data"/"chroma_naive_gemini"),
                 embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))
    raw=dense.get(include=["documents","metadatas"])
    documents=raw["documents"]
    metadatas=raw.get("metadatas") or [{} for _ in documents]
    assert len(documents)==len(metadatas)
    (root/"data"/"leaf_texts.json").write_text(json.dumps(documents,ensure_ascii=False))
    mapping={f"chunk_{i}":(meta or {}) for i,meta in enumerate(metadatas)}
    (root/"data"/"id_to_metadata.json").write_text(json.dumps(mapping,ensure_ascii=False))
    return len(documents)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    a=p.parse_args()
    print("Exported aligned chunks:",export(a.root))

if __name__=="__main__":main()

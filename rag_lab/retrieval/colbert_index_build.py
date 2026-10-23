"""Build an aligned PyLate/PLAID index; refuse accidental index overwrite.

1. python -m rag_lab.retrieval.export_colbert_leaves --root .
2. python -m rag_lab.retrieval.colbert_index_build --root . --batch-size 4
3. python -m rag_lab.retrieval.colbert_health --root . --query "PINN residual"

Use --overwrite ONLY when intentionally replacing an existing index.
CPU and small batches are recommended for 8GB RAM laptops.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from rag_lab.retrieval.colbert_backend import validate_mappings


def build(root, *, batch_size=4, overwrite=False, model_name="lightonai/colbertv2.0"):
    root=Path(root)
    data=root/"data"
    docs=json.loads((data/"leaf_texts.json").read_text(encoding="utf-8"))
    metadata=json.loads((data/"id_to_metadata.json").read_text(encoding="utf-8"))
    docids=[f"chunk_{i}" for i in range(len(docs))]
    texts=dict(zip(docids,docs))
    validate_mappings(texts,metadata)
    index_folder=data/"colbert_index"
    if index_folder.exists() and any(index_folder.iterdir()) and not overwrite:
        raise FileExistsError(f"ColBERT index exists: {index_folder}. Set --overwrite to rebuild")
    if batch_size<1: raise ValueError("batch_size must be positive")
    from pylate import indexes,models
    model=models.ColBERT(model_name_or_path=model_name)
    embeddings=model.encode(docs,batch_size=batch_size,is_query=False,show_progress_bar=True)
    index=indexes.PLAID(index_folder=str(index_folder),override=overwrite)
    index.add_documents(documents_ids=docids,documents_embeddings=embeddings)
    target=data/"id_to_text.json"
    target.write_text(json.dumps(texts,ensure_ascii=False),encoding="utf-8")
    return len(docs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--model-name",default="lightonai/colbertv2.0")
    a=p.parse_args()
    print("Indexed passages:",build(a.root,batch_size=a.batch_size,
                                   overwrite=a.overwrite,model_name=a.model_name))


if __name__=="__main__":main()

"""
Standalone ColBERT indexing script, run via whichever Python
environment PyLate installs cleanly into — NOT inside the main
Jupyter notebook. Run with the environment's explicit python path,
never a bare `python` command:

    ~/ragatouille_env/bin/python rag_lab/retrieval/colbert_index_build.py

(the venv name is stale from the earlier RAGatouille attempt — fine to
reuse if it's confirmed Python 3.11, or build a fresh ~/pylate_env the
same way. What matters is using the environment's own explicit binary
path, to avoid the earlier activation mixup where a bare `python`
picked up an unrelated 3.14 interpreter.)

Writes TWO files: the PLAID index itself, and id_to_text.json — the
main notebook environment needs the latter to map retrieval results
back to source text, since PyLate's index never stores text directly.
"""

import json
from pathlib import Path
from pylate import indexes, models

project_root = Path("/home/mohnish/Codes/rag-lab")  # adjust if needed

with open(project_root / "data" / "leaf_texts.json") as f:
    leaf_texts = json.load(f)

document_ids = [f"chunk_{i}" for i in range(len(leaf_texts))]
id_to_text = dict(zip(document_ids, leaf_texts))

model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0")
#model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0")

document_embeddings = model.encode(
    leaf_texts,
    batch_size=32,
    is_query=False,
    show_progress_bar=True,
)

index = indexes.PLAID(
    index_folder=str(project_root / "data" / "colbert_index"),
    override=True,
)
index.add_documents(
    documents_ids=document_ids,
    documents_embeddings=document_embeddings,
)

with open(project_root / "data" / "id_to_text.json", "w") as f:
    json.dump(id_to_text, f)

print(f"Indexed {len(leaf_texts)} chunks.")
print(f"Index: {project_root / 'data' / 'colbert_index'}")
print(f"id_to_text: {project_root / 'data' / 'id_to_text.json'}")
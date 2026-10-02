# colbert_index_build.py
import json
from pathlib import Path
from ragatouille import RAGPretrainedModel

project_root = Path("/home/mohnish/Codes/rag-lab") 

with open(project_root / "data" / "leaf_texts.json") as f:
    leaf_texts = json.load(f)

# Pass index_root inside from_pretrained()
colbert = RAGPretrainedModel.from_pretrained(
    "colbert-ir/colbertv2.0",
    index_root=str(project_root / "data" / "colbert_index")
)

index_path = colbert.index(
    collection=leaf_texts,
    index_name="rag_lab_colbert",
)

print("Index built at:", index_path)
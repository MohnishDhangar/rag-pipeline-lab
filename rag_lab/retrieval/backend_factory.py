"""Create local RAG retrieval backends without loading GPT/OAuth dependencies.

Dense Chroma imports the Gemini embedding client only when selected.
ColBERT/PyLate imports only its local backend when selected.
Importing this module alone never requires a GPT access token or PyJWT.
"""
import json
import os
from pathlib import Path


def create_backend(name, root, chroma_path=None, chroma_collection=None):
    if name=="dense":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_chroma import Chroma
        from rag_lab.retrieval.chroma_health import require_populated_index
        path = Path(chroma_path or os.environ.get("RAG_CHROMA_PATH")
                    or (root/"data"/"chroma_naive_gemini")).expanduser().resolve()
        collection = chroma_collection or os.environ.get("RAG_CHROMA_COLLECTION", "langchain")
        count = require_populated_index(path, collection)
        print(f"Using Chroma: {path}, collection={collection}, chunks={count}")
        embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return Chroma(embedding_function=embeddings,
                      persist_directory=str(path), collection_name=collection)
    from rag_lab.retrieval.colbert_backend import ColBERTVectorStoreShim
    with (root/"data"/"id_to_text.json").open() as f: mapping=json.load(f)
    metadata_path=root/"data"/"id_to_metadata.json"
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return ColBERTVectorStoreShim(index_folder=str(root/"data"/"colbert_index"),
                                  id_to_text=mapping, id_to_metadata=metadata)

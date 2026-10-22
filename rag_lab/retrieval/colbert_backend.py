"""
ColBERT retrieval backend, built on PyLate (github.com/lightonai/pylate)
rather than RAGatouille.

Switched from RAGatouille because its dependency chain (colbert-ai,
llama-index, faiss-cpu, voyager, onnx, plus a full CUDA toolkit
pull-in) failed to resolve on Python 3.14, even in a fresh isolated
venv — the "successful" RAGatouille install actually landed in an
unrelated pre-existing Python 3.11 environment, not the intended one,
and running the indexing script with a bare `python` picked up yet a
third interpreter with nothing installed. PyLate is built on the much
lighter Sentence Transformers stack, chosen specifically to avoid
repeating that dependency-resolution problem.

IMPORTANT — verified, not guessed: PyLate's add_documents() only ever
receives document IDs + embeddings, never raw text — so retrieve()
returns ids and scores, not text. This module maps ids back to source
text via `id_to_text`, built once at index-build time. Confirmed via
mock-based interface testing before this was written into the six
strategy files; still worth running the raw-result smoke test in the
notebook (below) against the REAL PyLate library once installed,
since mocks can't catch a genuine field-name mismatch in the actual
library.

ColBERTVectorStoreShim exposes only the two interface points every
existing strategy actually calls on a Chroma vectorstore:
  - .as_retriever(search_kwargs={"k": k}) -> object with .invoke(query)
    used by: naive, multi_query, rag_fusion, step_back, decomposition
  - .similarity_search(query, k=k) -> List[Document]
    used by: hyde (calls this directly, no as_retriever())
This isolates all six strategy files from knowing which ColBERT
library is in use — only this file changed when swapping libraries.
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document
from pylate import indexes, models, retrieve
from langchain_core.runnables import Runnable  # add this import


class ColBERTVectorStoreShim:
    def __init__(
        self,
        index_folder: str,
        id_to_text: Dict[str, str],
        #model_name: str = "colbert-ir/colbertv2.0",
        model_name: str = "lightonai/colbertv2.0",
    ):
        """
        id_to_text: maps each indexed chunk's id -> its original text.
        Required because PyLate's index returns ids, not text. Build
        this once in colbert_index_build.py and load it back here —
        do not rebuild the index just to get it.
        """
        self.model = models.ColBERT(model_name_or_path=model_name)
        self.index = indexes.PLAID(index_folder=index_folder, override=False)
        self.retriever = retrieve.ColBERT(index=self.index)
        self.id_to_text = id_to_text

    def _embed_query(self, query: str):
        return self.model.encode([query], is_query=True, show_progress_bar=False)

    def _results_to_documents(self, raw_results: list) -> List[Document]:
        docs = []
        for r in raw_results:
            doc_id = r.get("id")
            score = r.get("score")
            text = self.id_to_text.get(doc_id)
            if text is None:
                text = f"[WARNING: no text found for id={doc_id!r} — check id_to_text mapping]"
            docs.append(Document(page_content=text, metadata={"score": score, "id": doc_id}))
        return docs

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_embedding = self._embed_query(query)
        # PyLate returns one result-list PER query; [0] since we embed one at a time
        raw_results = self.retriever.retrieve(queries_embeddings=query_embedding, k=k)[0]
        return self._results_to_documents(raw_results)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        k = (search_kwargs or {}).get("k", 4)
        return _ShimRetriever(self, k)

class _ShimRetriever(Runnable):
    def __init__(self, shim: ColBERTVectorStoreShim, k: int):
        self.shim = shim
        self.k = k

    def invoke(self, input: str, config=None, **kwargs) -> List[Document]:
        return self.shim.similarity_search(input, k=self.k)

    
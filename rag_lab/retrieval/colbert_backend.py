"""PyLate/PLAID late-interaction backend, with strict original-text provenance.

Unlike dense Chroma scores, ColBERT MaxSim scores are not normalized to 0..1
and must not be compared directly with Chroma similarity scores.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
from langchain_core.documents import Document
from langchain_core.runnables import Runnable


def validate_mappings(id_to_text, id_to_metadata, *, require_metadata=True):
    if not id_to_text:
        raise ValueError("Empty ColBERT id_to_text mapping; build/export the index first")
    if set(id_to_text) != set(id_to_metadata) and require_metadata:
        missing=set(id_to_text)-set(id_to_metadata)
        extra=set(id_to_metadata)-set(id_to_text)
        raise ValueError(f"ColBERT metadata/text IDs differ: missing={len(missing)}, extra={len(extra)}")
    for key, text in id_to_text.items():
        if not isinstance(key,str) or not isinstance(text,str) or not text.strip():
            raise ValueError(f"Invalid ColBERT passage for ID {key!r}")
        meta=id_to_metadata.get(key,{})
        if not isinstance(meta,dict):
            raise ValueError(f"Invalid metadata for ColBERT ID {key!r}")
        if require_metadata and (not meta.get("source") or meta.get("page") is None):
            raise ValueError(f"ColBERT ID {key!r} lacks source/page provenance")


class ColBERTVectorStoreShim:
    def __init__(self, index_folder: str, id_to_text: Dict[str,str],
                 id_to_metadata: Optional[Dict[str,dict]]=None,
                 model_name: str="lightonai/colbertv2.0",
                 *, model=None, index=None, retriever=None,
                 require_metadata=True):
        self.index_folder=Path(index_folder)
        self.id_to_text=id_to_text
        self.id_to_metadata=id_to_metadata or {}
        validate_mappings(id_to_text,self.id_to_metadata,require_metadata=require_metadata)
        self._source_to_ids={}
        for chunk_id,meta in self.id_to_metadata.items():
            path=meta.get("source")
            if path:
                self._source_to_ids.setdefault(path,[]).append(chunk_id)
        # A verified index must have been encoded by the *same* model used
        # for query vectors. The manifest protects against accidental mismatch.
        from rag_lab.retrieval.colbert_manifest import validate_manifest
        manifest=validate_manifest(self.index_folder.parent.parent,
                                   id_to_text,self.id_to_metadata)
        if manifest["status"]=="matched" and manifest["model"]!=model_name:
            raise ValueError("ColBERT query model differs from indexed embedding model")
        if model is None or index is None or retriever is None:
            if not self.index_folder.is_dir() or not any(self.index_folder.iterdir()):
                raise FileNotFoundError(f"Missing ColBERT index: {self.index_folder}")
            try:
                from pylate import indexes, models, retrieve
            except ImportError as e:
                raise ImportError('Install ColBERT extras: pip install -e ".[colbert]"') from e
            model=model or models.ColBERT(model_name_or_path=model_name)
            index=index or indexes.PLAID(index_folder=str(self.index_folder),override=False)
            retriever=retriever or retrieve.ColBERT(index=index)
        self.model,self.index,self.retriever=model,index,retriever

    def _embed_query(self, query: str):
        return self.model.encode([query],is_query=True,show_progress_bar=False)

    def _results_to_documents(self, raw_results):
        docs=[]
        for rank,item in enumerate(raw_results,1):
            if not isinstance(item,dict) or "id" not in item:
                raise ValueError("PyLate must return dictionaries containing document IDs")
            doc_id=str(item["id"])
            if doc_id not in self.id_to_text:
                raise KeyError(f"ColBERT returned unknown chunk ID {doc_id!r}; index/mapping misaligned")
            score=item.get("score")
            meta={**self.id_to_metadata.get(doc_id,{}),
                  "id":doc_id,"original_id":doc_id,"score":score,
                  "retrieval_rank":rank,"retrieval_backend":"colbert"}
            docs.append(Document(page_content=self.id_to_text[doc_id],metadata=meta))
        return docs

    def similarity_search(self, query: str, k: int=4, **kwargs):
        if k<1: raise ValueError("k must be positive")
        filters=kwargs.pop("filter",None)
        if kwargs:
            raise ValueError(f"Unsupported ColBERT search options: {sorted(kwargs)}")
        params={"queries_embeddings":self._embed_query(query),"k":k}
        allowed=None
        if filters is not None:
            if not isinstance(filters,dict) or set(filters)!={"source"} or not isinstance(filters["source"],str):
                raise ValueError("ColBERT supports only exact source-path filtering")
            allowed=self._source_to_ids.get(filters["source"],[])
            if not allowed:
                return []
            # PyLate's documented subset argument runs PLAID only against
            # the permitted indexed chunk IDs, unlike global top-k filtering.
            params["subset"]=allowed
        output=self.retriever.retrieve(**params)
        if not isinstance(output,(list,tuple)) or len(output)!=1:
            raise ValueError("PyLate must return one result-list for the single query")
        docs=self._results_to_documents(output[0])
        if allowed is not None and any(d.metadata["original_id"] not in set(allowed) for d in docs):
            raise ValueError("PyLate subset filtering returned a document outside its source filter")
        return docs

    def as_retriever(self, search_kwargs=None, search_type="similarity"):
        if search_type!="similarity":
            raise ValueError("ColBERT backend supports similarity/MaxSim only; not dense MMR")
        return _ShimRetriever(self,(search_kwargs or {}).get("k",4))


class _ShimRetriever(Runnable):
    def __init__(self, shim, k):
        self.shim,self.k=shim,k
    def invoke(self,input,config=None,**kwargs):
        return self.shim.similarity_search(input,k=self.k)

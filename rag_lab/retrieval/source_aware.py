"""Text-only requested-paper detection and bounded, local source-targeted retrieval.

Gold benchmark question_type/source_documents are NEVER consulted.
When the question depends on unknown prior context ('these three papers'),
return an unresolved-scope warning rather than guessing papers.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from collections import defaultdict

EXPLICIT_NAMES = {
    "ntk_pinn": ("NTK-PINN", "NTK PINN", "neural tangent kernel paper"),
    "xpinn": ("XPINN", "XPINNs", "extended physics-informed neural network"),
    "sk_pinn": ("SK-PINN", "SK PINN", "smoothing kernel gradient"),
    "gns": ("GNS", "graph network simulator", "learning to simulate complex physics"),
    "meshgraphnets": ("MeshGraphNets", "mesh graph nets", "mesh-based simulation with graph networks"),
    "pignn_al": ("PIGNN active-learning", "PIGNN active learning", "physics-informed graph neural network active learning"),
    "pignn_fd": ("finite-difference-based PIGNN", "PIGNN finite-difference", "PIGNN-FD", "PIGNN and finite difference"),
    "pignn_accel": ("accelerated multi-GNN", "accelerated PIGNN", "multi-GNN PIGNN"),
    "dist_pinn_dd": ("distributed PINN", "distributed physics-informed neural network", "distributed PINNs"),
    "pinn_gnn_pue": ("PINN-GNN PUE", "PINN-GNN hybrid", "PUE prediction"),
    "orig_pinn": ("Raissi et al.", "original PINNs paper", "original PINN paper"),
}
SCOPE_REFERENCES = re.compile(
    r"\b(?:all three papers|three papers in this batch|this batch|these papers|"
    r"those papers|this paper(?:'s)?|the two papers)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SourceRequest:
    paper_ids: tuple[str, ...]
    unresolved_scope: bool
    explicit_matches: tuple[str, ...]


def requested_sources(question: str, registry=None) -> SourceRequest:
    text=" ".join(str(question).split())
    found=[]
    matching=[]
    for paper_id,aliases in EXPLICIT_NAMES.items():
        if registry is not None and paper_id not in registry.paper_ids:
            continue
        for alias in aliases:
            rx=r"(?<![a-zA-Z0-9])"+re.escape(alias)+r"(?![a-zA-Z0-9])"
            if re.search(rx,text,re.IGNORECASE):
                if paper_id not in found:found.append(paper_id)
                matching.append(alias)
                break
    # Do not assume which external paper 'this paper' or 'this batch'
    # denotes. Users may supply this information explicitly via a source list
    # in a future multi-turn interface; evaluation labels are not that list.
    scope=bool(SCOPE_REFERENCES.search(text))
    return SourceRequest(tuple(found),scope and len(found)<2,tuple(matching))


def document_key(doc):
    meta=getattr(doc,"metadata",{})
    return (meta.get("source"),meta.get("page"),
            getattr(doc,"page_content",""))


class SourceAwareRetriever:
    def __init__(self, vectorstore, registry, *, k=4, pool_k=28, max_target_queries=3):
        self.store,self.registry=vectorstore,registry
        self.k,self.pool_k=k,pool_k
        self.max_target_queries=max_target_queries
        self._paths=None

    def _base(self):
        return getattr(self.store,"base",self.store)

    def _sources_by_paper(self):
        if self._paths is not None: return self._paths
        paths=defaultdict(set)
        base=self._base()
        if hasattr(base,"get"):
            # Chroma stores original 'source' metadata and supports exact
            # source-path filters, which can avoid losing rare papers in top-k.
            raw=base.get(include=["metadatas"])
            metas=raw.get("metadatas",[]) or []
        else:
            metas=getattr(base,"id_to_metadata",{}).values()
        for meta in metas:
            source=(meta or {}).get("source")
            paper=self.registry.resolve(source)
            if paper and source: paths[paper].add(source)
        self._paths=paths
        return paths

    def _search(self, query, limit, paper=None):
        paths=self._sources_by_paper().get(paper,set()) if paper else set()
        if paper and paths and hasattr(self._base(),"get"):
            docs=[]
            for path in sorted(paths):
                # Chroma where={"source": exact path}; no gold labels.
                docs.extend(self.store.similarity_search(query,k=limit,filter={"source":path}))
            return [d for d in docs if self.registry.resolve(d.metadata.get("source"))==paper][:limit]
        docs=self.store.similarity_search(query,k=limit)
        if paper:
            docs=[d for d in docs if self.registry.resolve(d.metadata.get("source"))==paper]
        return docs

    def retrieve(self, question, *, requested=None, initial=None):
        request=requested or requested_sources(question,self.registry)
        candidates=list(initial) if initial is not None else self._search(question,self.k)
        by_paper={self.registry.resolve(d.metadata.get("source")) for d in candidates}
        target_ids=list(request.paper_ids)
        quota=max(1,self.k//max(1,len(target_ids)))
        # Only explicitly identified papers receive targeted searches.
        retries=0
        for paper in target_ids:
            current=sum(self.registry.resolve(d.metadata.get("source"))==paper for d in candidates)
            if current>=quota: continue
            if retries>=self.max_target_queries:break
            retries+=1
            query=f"{paper.replace('_',' ')} {question}"
            docs=self._search(query,self.pool_k,paper=paper)
            candidates.extend(docs[:max(quota-current,1)])
        # Preserve at least one passage per explicit source if available;
        # fill remaining slots from initial rank order.
        chosen=[]
        used=set()
        for paper in target_ids:
            for d in candidates:
                key=document_key(d)
                if key not in used and self.registry.resolve(d.metadata.get("source"))==paper:
                    chosen.append(d);used.add(key);break
        for d in candidates:
            key=document_key(d)
            if key not in used and len(chosen)<self.k:
                chosen.append(d);used.add(key)
        covered={self.registry.resolve(d.metadata.get("source")) for d in chosen}
        missing=[paper for paper in target_ids if paper not in covered]
        return chosen[:self.k],{
            "requested_papers":target_ids,
            "matched_aliases":list(request.explicit_matches),
            "unresolved_scope":request.unresolved_scope,
            "missing_requested_papers":missing,
            "targeted_retrieval_queries":retries,
        }

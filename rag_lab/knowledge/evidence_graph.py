"""Offline, provenance-preserving paper/concept/claim graph.

Deterministic construction avoids GPT ingestion costs. A keyword link is an
association, not proof of causation or scientific truth.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib, json, re

CONCEPTS={
    "pinn":r"\b(pinns?|physics.informed neural networks?)\b",
    "gnn":r"\b(gnns?|graph neural networks?)\b",
    "pde":r"\b(pdes?|partial differential equations?)\b",
    "mesh":r"\b(meshes|mesh.based|unstructured mesh)\b",
    "domain_decomposition":r"\b(domain decomposition|subdomains?|xpinns?)\b",
    "fluid_simulation":r"\b(fluid (flow|simulation|dynamics)|navier.stokes)\b",
    "training":r"\b(training|optimization|gradients?)\b",
}
def concepts_in(text):
    return {name for name,expr in CONCEPTS.items() if re.search(expr,text,re.I)}

@dataclass
class Evidence:
    id: str
    paper_id: str
    page: int | None
    text: str
    source: str | None
    concepts: list[str]

class EvidenceGraph:
    def __init__(self):
        self.papers={}
        self.nodes={}
        self.edges=[]

    def add_paper(self, paper_id, title="", source=None):
        self.papers[paper_id]={"title":title,"source":source}

    def add_passage(self, paper_id, text, page=None, source=None):
        if not paper_id or not text.strip():
            return
        eid=hashlib.sha256((str(paper_id)+":"+str(page)+":"+text).encode()).hexdigest()[:20]
        if eid in self.nodes: return
        names=sorted(concepts_in(text))
        self.nodes[eid]=Evidence(eid,paper_id,page,text,source,names)
        self.edges.append([paper_id,eid,"contains"])
        for concept in names:
            self.edges.append([eid,"concept:"+concept,"mentions"])

    def query(self, question, k=4):
        wanted=concepts_in(question)
        terms=set(re.findall(r"[a-z]{4,}",question.lower()))
        scored=[]
        for node in self.nodes.values():
            overlap=len(wanted.intersection(node.concepts))
            lexical=len(terms.intersection(set(re.findall(r"[a-z]{4,}",node.text.lower()))))
            if overlap or lexical:
                scored.append((3*overlap+lexical,node))
        scored.sort(key=lambda item:item[0],reverse=True)
        return [node for _,node in scored[:k]]

    def to_dict(self):
        return {"papers":self.papers,
                "nodes":{k:vars(v) for k,v in self.nodes.items()},
                "edges":self.edges}

    @classmethod
    def from_dict(cls,data):
        g=cls()
        g.papers=data.get("papers",{})
        g.nodes={k:Evidence(**v) for k,v in data.get("nodes",{}).items()}
        g.edges=data.get("edges",[])
        return g

    def save(self,path):
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False))

    @classmethod
    def load(cls,path):
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

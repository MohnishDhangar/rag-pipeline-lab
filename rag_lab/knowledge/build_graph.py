"""Build evidence graph from existing paper metadata and local PDFs.

python -m rag_lab.knowledge.build_graph --papers data/papers --eval eval/eval_set.json
"""
import argparse, json
from pathlib import Path
from .evidence_graph import EvidenceGraph

def build(paper_dir, evaluation_path, output):
    from pypdf import PdfReader
    base=Path(paper_dir)
    metadata=json.loads(Path(evaluation_path).read_text())["metadata"]["papers"]
    g=EvidenceGraph()
    count=0
    for paper in metadata:
        short_id=paper["short_id"]
        g.add_paper(short_id,paper["title"],paper.get("file"))
        filename=paper.get("file")
        if not filename or not (base/filename).exists():
            continue
        reader=PdfReader(str(base/filename))
        for i,page in enumerate(reader.pages,1):
            text=page.extract_text() or ""
            paragraphs=[p.strip() for p in text.split("\n\n") if len(p.strip())>60]
            if not paragraphs and text.strip(): paragraphs=[text.strip()]
            for paragraph in paragraphs:
                g.add_passage(short_id,paragraph,page=i,source=filename)
                count+=1
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    g.save(output)
    return {"papers":len(g.papers),"evidence_nodes":len(g.nodes),"passages_processed":count}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--papers",default="data/papers")
    p.add_argument("--eval",default="eval/eval_set.json")
    p.add_argument("--output",default="data/evidence_graph.json")
    a=p.parse_args()
    print(build(a.papers,a.eval,a.output))

if __name__=="__main__":main()

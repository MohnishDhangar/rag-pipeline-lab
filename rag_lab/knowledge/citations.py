"""Pass-through source labels and deterministic citation ID validation."""
import re
from rag_lab.core.trace import source_record

def format_context(doc):
    source=source_record(doc)
    ident=source["id"]
    page=source["page"]
    locator=ident+(":page"+str(page) if page is not None else "")
    return "[SOURCE "+locator+"]\n"+getattr(doc,"page_content","")

def citation_coverage(answer, sources):
    """ID validity only; semantic faithfulness needs further evaluation."""
    allowed={str(s.get("id")) for s in sources if s.get("id")}
    found=re.findall(r"\[SOURCE\s+([^\]]+)\]",answer)
    valid=sum(c.split(":page")[0] in allowed for c in found)
    return {"citations":len(found),"valid_citations":valid,
            "validity":valid/len(found) if found else None}

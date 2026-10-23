"""Citation provenance and evidence checks, without extra GPT calls."""
import re
from pathlib import Path

def format_context(doc):
    meta=getattr(doc,"metadata",{}) or {}
    ref=meta.get("id") or Path(str(meta.get("source","unknown"))).name
    page=meta.get("page")
    locator=str(ref)+(":page"+str(page) if page is not None else "")
    return "[SOURCE "+locator+"]\n"+getattr(doc,"page_content","")

def citation_coverage(answer, sources):
    """Lexical citation-ID validity, not semantic faithfulness."""
    allowed=set(str(s.get("id")) for s in sources if s.get("id"))
    citations=re.findall(r"\[SOURCE\s+([^\]]+)\]",answer)
    matched=sum(c.split(":page")[0] in allowed for c in citations)
    return {"citations":len(citations),"valid_citations":matched,
            "validity":matched/len(citations) if citations else None}

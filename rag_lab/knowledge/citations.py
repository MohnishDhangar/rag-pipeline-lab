"""Exact source labels and deterministic citation validation.

PDF page numbers start at 1 in new citations. Old zero-based references from
past benchmark runs remain checkable through legacy_id and source metadata.
This only validates provenance IDs, never semantic support of a claim.
"""
import re
from rag_lab.core.trace import source_record


def citation_label(source):
    ident = str(source["id"])
    n = source.get("pdf_page_number")
    if n is not None:
        return f"{ident}:pdfpage{n}"
    page = source.get("page")
    return f"{ident}:page{page}" if page is not None else ident


def format_context(doc):
    src = source_record(doc)
    return "[SOURCE "+citation_label(src)+"]\n"+getattr(doc,"page_content","")


def citation_coverage(answer, sources):
    accepted=set()
    for s in sources:
        if not s.get("id"): continue
        accepted.add(citation_label(s))
        # Legacy source labels used PDF filenames and zero-based page indices.
        for original in (s.get("legacy_id"),):
            if original:
                page=s.get("page")
                accepted.add(f"{original}:page{page}" if page is not None else original)
    found=re.findall(r"\[SOURCE\s+([^\]]+)\]",answer or "")
    valid=sum(label.strip() in accepted for label in found)
    return {"citations":len(found), "valid_citations":valid,
            "validity":valid/len(found) if found else None}

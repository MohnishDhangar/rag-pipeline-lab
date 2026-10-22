"""
CRAG -- Corrective Retrieval Augmented Generation.

Implements the correction layer from Yan et al., "Corrective Retrieval
Augmented Generation" (2024): a lightweight relevance grader sits between
retrieval and generation, and routes each query down one of three paths
depending on how much it trusts what came back.

  CORRECT   -- at least one retrieved doc is relevant and none are
               irrelevant. The correct doc(s) are "refined" (decomposed
               into knowledge strips, filtered to the relevant ones,
               recomposed) and used as context, same as any other
               strategy here.
  INCORRECT -- every retrieved doc is irrelevant. The retrieved set is
               discarded entirely and a web search is substituted in its
               place.
  AMBIGUOUS -- a mixed verdict (some correct, some incorrect, or any
               doc graded ambiguous). Refined internal knowledge and a
               web search are combined.

This mirrors the paper's three actions directly, rather than the common
simplified "binary relevant/not -> corrective retrieval" version of CRAG
seen in some tutorials -- the point of grading into three buckets instead
of two is exactly to have a distinct AMBIGUOUS path.

Deliberate simplifications, called out so they don't read as bugs later:

  - Knowledge refinement (decompose -> filter -> recompose) is done as a
    SINGLE structured-output call per correct document, asking the grader
    LLM to extract only the relevant strips directly, rather than three
    separate calls (decompose into strips, grade each strip, recompose).
    Same end result, one call instead of N+2 -- a deliberate concession to
    Groq's free-tier daily token cap, which this project has hit hard and
    repeatedly in earlier phases.
  - Query rewriting before web search is a single plain (non-structured)
    LLM call, only triggered on the INCORRECT/AMBIGUOUS paths -- not on
    every query -- for the same reason.

Interface matches every other strategy class in rag_lab/strategies/:
  CRAGStrategy(vectorstore, llm=llm).run(query) -> str

It also exposes `.run_with_trace(query) -> dict`, returning the grades,
the path taken, and the web-search query/results if any -- useful for
demoing the correction behavior in a notebook, and for the eval harness
if CRAG is ever added as a seventh strategy there.
"""

from typing import Callable, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured-output schemas for the grader
# ---------------------------------------------------------------------------

class RelevanceGrade(BaseModel):
    """Verdict for a single retrieved document against the query."""
    grade: Literal["correct", "incorrect", "ambiguous"] = Field(
        description=(
            "'correct' if the document clearly contains information that "
            "answers or is directly relevant to the question. 'incorrect' "
            "if it is unrelated. 'ambiguous' if it's partially relevant, "
            "touches the right topic without answering the question, or "
            "you're not confident either way."
        )
    )
    reason: str = Field(description="One short sentence explaining the grade.")


class RefinedKnowledge(BaseModel):
    """Output of knowledge refinement: the document reduced to only the
    strips (sentences/clauses) relevant to the query."""
    relevant_strips: List[str] = Field(
        description=(
            "The sentences or clauses from the document that are directly "
            "relevant to answering the question, copied verbatim. Omit "
            "boilerplate, citations, and unrelated sentences. If nothing "
            "is relevant, return an empty list."
        )
    )


GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict grader assessing whether a retrieved document is \
relevant to a user's question.

Question: {question}

Retrieved document:
{document}

Grade this document's relevance to the question."""
)

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """Extract only the sentences from this document that are directly \
relevant to answering the question. Discard everything else -- citations, \
boilerplate, unrelated discussion.

Question: {question}

Document:
{document}"""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the following question as a short, keyword-focused web \
search query (no more than 12 words). Return ONLY the rewritten query, \
nothing else.

Question: {question}"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}"""
)


# ---------------------------------------------------------------------------
# Web search fallback
# ---------------------------------------------------------------------------

def default_web_search(query: str, max_results: int = 4) -> List[Document]:
    """Keyless web search via duckduckgo-search (import name `ddgs` in
    recent releases, `duckduckgo_search` in older ones -- both are tried).

    Degrades gracefully: if the package isn't installed, or the sandbox
    has no outbound access to DuckDuckGo, this returns an empty list with
    a single placeholder Document explaining why, instead of raising --
    a hard crash here would take down the whole CORRECTIVE path, which
    defeats the point of having a fallback in the first place.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # older package name

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return [Document(
                page_content=f"[web search for {query!r} returned no results]",
                metadata={"source": "web_search", "query": query},
            )]
        return [
            Document(
                page_content=f"{r.get('title', '')}\n{r.get('body', '')}",
                metadata={"source": "web_search", "url": r.get("href"), "query": query},
            )
            for r in results
        ]
    except Exception as e:
        return [Document(
            page_content=(
                f"[web search unavailable ({type(e).__name__}: {e}) -- "
                f"falling back to internal knowledge only for: {query!r}]"
            ),
            metadata={"source": "web_search_failed", "query": query},
        )]


# ---------------------------------------------------------------------------
# CRAGStrategy
# ---------------------------------------------------------------------------

class CRAGStrategy:
    def __init__(
        self,
        vectorstore,
        llm,
        grader_llm=None,
        k: int = 4,
        web_search_fn: Optional[Callable[[str], List[Document]]] = None,
        verbose: bool = False,
    ):
        """
        vectorstore: anything exposing .as_retriever(search_kwargs={"k": k})
                     -> object with .invoke(query) -> List[Document].
                     Works unmodified with the dense Chroma store AND the
                     ColBERTVectorStoreShim -- same contract every other
                     strategy in this package relies on.
        llm:         used for final answer generation.
        grader_llm:  used for grading/refinement/rewrite calls. Defaults to
                     `llm`. Pass a separate (e.g. smaller/local) model here
                     to keep the grading calls off whichever provider's
                     quota is tightest -- grading is 1 call per retrieved
                     doc, so it adds up fast on a k=4-6 retriever.
        web_search_fn: query:str -> List[Document]. Defaults to
                     `default_web_search` (DuckDuckGo, keyless).
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.grader_llm = grader_llm or llm
        self.k = k
        self.web_search_fn = web_search_fn or default_web_search
        self.verbose = verbose

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self._grader = self.grader_llm.with_structured_output(RelevanceGrade)
        self._refiner = self.grader_llm.with_structured_output(RefinedKnowledge)
        self._rewriter = REWRITE_PROMPT | self.grader_llm | StrOutputParser()
        self._answerer = ANSWER_PROMPT | self.llm | StrOutputParser()

    # -- steps -----------------------------------------------------------

    def _grade_document(self, query: str, doc: Document) -> RelevanceGrade:
        chain = GRADE_PROMPT | self._grader
        return chain.invoke({"question": query, "document": doc.page_content})

    def _refine_document(self, query: str, doc: Document) -> str:
        chain = REFINE_PROMPT | self._refiner
        refined = chain.invoke({"question": query, "document": doc.page_content})
        if not refined.relevant_strips:
            return ""
        return " ".join(refined.relevant_strips)

    def _rewrite_for_search(self, query: str) -> str:
        return self._rewriter.invoke({"question": query}).strip()

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # -- main entry point --------------------------------------------------

    def run_with_trace(self, query: str) -> dict:
        trace = {"query": query, "grades": [], "path": None,
                 "web_search_query": None, "web_results": [], "answer": None}

        docs = self.retriever.invoke(query)
        grades = [self._grade_document(query, d) for d in docs]
        for doc, g in zip(docs, grades):
            trace["grades"].append({
                "preview": doc.page_content[:120],
                "grade": g.grade,
                "reason": g.reason,
            })
        self._log(f"[crag] graded {len(docs)} docs: {[g.grade for g in grades]}")

        verdicts = {g.grade for g in grades}
        has_correct = "correct" in verdicts
        has_incorrect = "incorrect" in verdicts
        has_ambiguous = "ambiguous" in verdicts

        # Truth table (deliberately spelled out rather than compressed):
        #   all correct                      -> correct    (trust internal knowledge alone)
        #   all incorrect                     -> incorrect  (discard, web search only)
        #   anything else (mixed, or any      -> ambiguous  (combine refined internal +
        #     doc graded ambiguous)                          web search)
        if has_correct and not has_incorrect and not has_ambiguous:
            path = "correct"
        elif has_incorrect and not has_correct and not has_ambiguous:
            path = "incorrect"
        else:
            path = "ambiguous"
        trace["path"] = path

        internal_context = ""
        if path in ("correct", "ambiguous"):
            correct_docs = [d for d, g in zip(docs, grades) if g.grade in ("correct", "ambiguous")]
            refined = [self._refine_document(query, d) for d in correct_docs]
            refined = [r for r in refined if r]
            internal_context = "\n".join(refined)
            self._log(f"[crag] refined {len(refined)}/{len(correct_docs)} docs into knowledge strips")

        web_context = ""
        if path in ("incorrect", "ambiguous"):
            search_query = self._rewrite_for_search(query)
            trace["web_search_query"] = search_query
            web_docs = self.web_search_fn(search_query)
            trace["web_results"] = [
                {"preview": d.page_content[:150], "metadata": d.metadata} for d in web_docs
            ]
            web_context = "\n".join(d.page_content for d in web_docs)
            self._log(f"[crag] web search ({search_query!r}) -> {len(web_docs)} results")

        context = "\n\n".join(c for c in (internal_context, web_context) if c)
        if not context:
            context = "[no relevant information found internally or via web search]"
        trace["context_used"] = context

        trace["answer"] = self._answerer.invoke({"context": context, "question": query})
        return trace

    def run(self, query: str) -> str:
        return self.run_with_trace(query)["answer"]

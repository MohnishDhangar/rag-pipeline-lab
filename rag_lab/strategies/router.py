"""
LogicalRouter -- classifies an incoming question into one of the 8
strategies built in this project (naive, multi_query, rag_fusion, hyde,
step_back, decomposition, colbert, crag), then runs the chosen strategy
and returns its result.

This is your own router.py, unchanged in design, with exactly one
addition: `choose()` now wraps the routing LLM call in
`call_with_backoff` (from rag_lab.utils), the same retry/quota-detection
helper every other LLM call in this project already goes through. The
router makes a real structured-output call every time `choose()` or
`run()` is invoked, so it's just as exposed to a transient 429/500 or a
hard TPD wall as generation or scoring is -- there's no reason for it to
be the one call site in the project that doesn't retry.

Routing here is LOGICAL, not semantic: an LLM classifies the question via
structured output into one of 8 Literal categories, rather than nearest-
neighbor embedding similarity -- the 8 strategies aren't different
topical domains to embed and match against, they're different RETRIEVAL
TECHNIQUES for the SAME corpus. Picking the right technique is a
reasoning judgment, which is what logical routing is suited for.

IMPORTANT -- the category `description` fields below are REASONED
DEFAULTS, not yet validated against this project's actual
eval_results.jsonl / scoring numbers. The companion notebook
(12_router.ipynb) includes a data-driven validation step against your
real per-(strategy, question_type) scores specifically to close this gap
-- read its output before trusting this in anything resembling
production, and edit the descriptions below to match what your data
actually shows.

Exposes the same .retrieve()/.run() interface as every other strategy.
One caveat worth repeating from the original docstring: routing TO
naive/multi_query/etc FROM an eval loop that's already comparing those
same strategies independently is circular -- the router's natural home
is a single "production" entry point, not a 9th row competing against
the 8 strategies it internally dispatches to.

Save this file as rag_lab/strategies/router.py.
"""

from typing import Dict, List, Literal, Optional, Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_lab.utils import call_with_backoff

StrategyName = Literal[
    "naive",
    "multi_query",
    "rag_fusion",
    "hyde",
    "step_back",
    "decomposition",
    "colbert",
    "crag",
]


class _RunsAndRetrieves(Protocol):
    """Structural type matching every strategy class in this project --
    not imported directly to avoid this module depending on all 8
    strategy modules just for a type hint."""

    def retrieve(self, query: str) -> List[Document]: ...
    def run(self, query: str) -> str: ...


class RouteQuery(BaseModel):
    """Route a user question to the most suitable RAG strategy for a
    corpus of physics-informed neural network (PINN) and graph neural
    network (GNN) research papers."""

    strategy: StrategyName = Field(
        ...,
        description=(
            "Pick exactly one:\n"
            "- naive: a simple, narrow, single-fact question, directly "
            "answerable from one passage, phrased close to how the "
            "corpus itself would state it.\n"
            "- multi_query: the question is ambiguous, vaguely phrased, "
            "or could reasonably be interpreted a few different ways -- "
            "rephrasing it several ways would help surface the right chunk.\n"
            "- rag_fusion: similar situation to multi_query, but when "
            "you want the single MOST robust answer via consensus "
            "ranking across phrasings rather than just broader recall -- "
            "prefer this over multi_query for borderline or noisy questions.\n"
            "- hyde: the question is phrased very differently "
            "(colloquially, abstractly, non-technically) from how a "
            "technical paper would state the answer, so a direct "
            "embedding match between question and passage is unlikely "
            "to work well.\n"
            "- step_back: the question is VERY narrow and specific (a "
            "single number, parameter, or detail attributed to one "
            "paper) and would benefit from first retrieving broader "
            "conceptual background before answering precisely.\n"
            "- decomposition: the question has multiple distinct parts, "
            "requires comparing or connecting findings ACROSS more than "
            "one paper, or needs multi-step reasoning to answer fully.\n"
            "- colbert: the question hinges on precise terminology, a "
            "specific named method, author, or acronym, or rare "
            "technical terms where exact wording matters more than "
            "general semantic similarity.\n"
            "- crag: the question might not be answerable from this "
            "corpus at all -- e.g. it asks about something current or "
            "external to these papers (a tool/library version, a recent "
            "event), or a topic outside physics-informed ML/GNNs -- and "
            "needs a safety net that can fall back to web search."
        ),
    )


class LogicalRouter:
    def __init__(
        self,
        strategies: Dict[StrategyName, _RunsAndRetrieves],
        router_llm: BaseChatModel,
    ):
        """
        strategies: dict covering ALL 8 StrategyName keys, mapped to
        already-constructed strategy instances (reuse the same
        instances your generation loop builds -- don't reconstruct new
        ones here, or you'll pay for duplicate LLM/embedding clients).
        router_llm: a tool-calling-capable model for the classification
        step. Same reasoning as crag.py's grader_llm: don't assume your
        generation llm supports with_structured_output reliably.
        """
        expected = set(RouteQuery.model_fields["strategy"].annotation.__args__)
        missing = expected - set(strategies)
        if missing:
            raise ValueError(f"strategies dict is missing required keys: {missing}")

        self.strategies = strategies
        self.structured_router_llm = router_llm.with_structured_output(RouteQuery)
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert at routing a user question to the most "
             "suitable retrieval-augmented generation strategy for a "
             "corpus of physics-informed neural network (PINN) and "
             "graph neural network (GNN) research papers."),
            ("human", "{question}"),
        ])
        self.router_chain = self.router_prompt | self.structured_router_llm
        self.last_route: Optional[str] = None  # set by choose(); handy for logging/inspection

    def choose(self, query: str) -> StrategyName:
        result = call_with_backoff(lambda: self.router_chain.invoke({"question": query}))
        self.last_route = result.strategy
        return result.strategy

    def retrieve(self, query: str) -> List[Document]:
        chosen = self.choose(query)
        return self.strategies[chosen].retrieve(query)

    def run(self, query: str) -> str:
        chosen = self.choose(query)
        return self.strategies[chosen].run(query)

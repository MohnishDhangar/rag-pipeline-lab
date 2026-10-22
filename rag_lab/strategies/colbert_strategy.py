"""
ColBERTStrategy — wraps ColBERTVectorStoreShim behind the same
.retrieve()/.run() interface every other strategy in rag_lab.strategies
exposes (see naive.py, multi_query.py, etc.), so ColBERT retrieval
drops straight into the eval harness's `strategies` dict in
generation.ipynb alongside naive/multi_query/rag_fusion/hyde/
step_back/decomposition.

Deliberately thin: ColBERTVectorStoreShim already exposes
.as_retriever(search_kwargs={"k": k}), mirroring a Chroma vectorstore's
interface (see colbert_backend.py's own docstring — this was the
explicit design intent of the shim). All this class adds is the
generation half (prompt + llm), matching NaiveStrategy's answer
prompt exactly, so any quality delta measured in eval_results.jsonl
is attributable to the retriever (dense vs. ColBERT), not to a
different prompt.

Save this file as rag_lab/strategies/colbert_strategy.py.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_lab.retrieval.colbert_backend import ColBERTVectorStoreShim


class ColBERTStrategy:
    """Naive RAG (single retrieval, no query translation) using the
    ColBERT/PyLate backend instead of dense Chroma retrieval."""

    def __init__(
        self,
        colbert_vectorstore: ColBERTVectorStoreShim,
        llm: BaseChatModel,
        k: int = 4,
    ):
        self.colbert_vectorstore = colbert_vectorstore
        self.retriever = colbert_vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}"
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    def run(self, query: str) -> str:
        docs = self.retrieve(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

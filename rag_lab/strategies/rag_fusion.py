from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from rag_lab.strategies.multi_query import LineListOutputParser, MULTI_QUERY_PROMPT


def reciprocal_rank_fusion(
    doc_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_lookup[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    reranked_keys = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
    return [doc_lookup[key] for key in reranked_keys]


class RagFusionStrategy:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, k: int = 4, rrf_k: int = 60):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.rrf_k = rrf_k
        self.query_gen_chain = MULTI_QUERY_PROMPT | llm | LineListOutputParser()
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}"
        )

    def generate_queries(self, query: str) -> List[str]:
        return [query] + self.query_gen_chain.invoke({"question": query})

    def retrieve(self, query: str) -> List[Document]:
        all_queries = self.generate_queries(query)
        doc_lists = [self.retriever.invoke(q) for q in all_queries]
        return reciprocal_rank_fusion(doc_lists, k=self.rrf_k)

    def run(self, query: str) -> str:
        docs = self.retrieve(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})
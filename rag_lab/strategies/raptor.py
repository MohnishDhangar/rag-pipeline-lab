from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore


class RaptorStrategy:
    """Queries a pre-built RAPTOR tree (leaf chunks + recursive cluster
    summaries at every level), all stored flat in one Chroma collection.
    Retrieval naturally selects whichever level's embedding is closest
    to the query — a specific question tends to match a leaf, a broad
    conceptual question tends to match a higher-level summary."""

    def __init__(self, raptor_vectorstore: VectorStore, llm: BaseChatModel, k: int = 4):
        self.retriever = raptor_vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context, "
            "which may include both specific passages and higher-level "
            "summaries:\n{context}\n\nQuestion: {question}"
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    def run(self, query: str) -> str:
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[Level {d.metadata.get('level', '?')}] {d.page_content}"
            for d in docs
        )
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})
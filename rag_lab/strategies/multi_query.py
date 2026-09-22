# rag_lab/strategies/multi_query.py
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return [line.strip() for line in text.strip().split("\n") if line.strip()]


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant. Generate 4 different rephrasings of the "
    "user question below, to help retrieve relevant documents from a "
    "vector database. Provide these alternative questions separated "
    "by newlines, with no numbering or extra commentary.\n\n"
    "Original question: {question}"
)


class MultiQueryStrategy:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, k: int = 4):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.query_gen_chain = MULTI_QUERY_PROMPT | llm | LineListOutputParser()
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}"
        )

    def generate_queries(self, query: str) -> List[str]:
        return [query] + self.query_gen_chain.invoke({"question": query})

    @staticmethod
    def _unique_union(doc_lists: List[List[Document]]) -> List[Document]:
        seen, unique_docs = set(), []
        for docs in doc_lists:
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
        return unique_docs

    def retrieve(self, query: str) -> List[Document]:
        all_queries = self.generate_queries(query)
        doc_lists = [self.retriever.invoke(q) for q in all_queries]
        return self._unique_union(doc_lists)

    def run(self, query: str) -> str:
        docs = self.retrieve(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})
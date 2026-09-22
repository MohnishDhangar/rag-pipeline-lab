from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from rag_lab.utils import call_with_backoff, normalize_text

HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Please write a passage that could plausibly appear in a technical "
    "paper, answering the question below. Write only the passage itself "
    "— no preamble, no meta-commentary, no acknowledgment that this is "
    "hypothetical.\n\n"
    "Question: {question}\n\n"
    "Passage:"
)


class HydeStrategy:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, k: int = 4):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.hyde_chain = HYDE_PROMPT | llm | StrOutputParser()
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}"
        )

    def generate_hypothetical_doc(self, query: str) -> str:
        return normalize_text(self.hyde_chain.invoke({"question": query}))
        
    def retrieve(self, query: str) -> List[Document]:
        #hypothetical_doc = self.generate_hypothetical_doc(query)
        #return self.vectorstore.similarity_search(hypothetical_doc, k=self.k)
        hypothetical_doc = self.generate_hypothetical_doc(query)
        return call_with_backoff(
            lambda: self.vectorstore.similarity_search(hypothetical_doc, k=self.k)
        )
        
    def run(self, query: str) -> str:
        docs = self.retrieve(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})
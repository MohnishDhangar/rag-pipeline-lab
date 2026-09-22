from langchain_core.language_models import BaseChatModel
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore


class NaiveStrategy:
    """Baseline RAG strategy: plain similarity retrieval, no query transformation."""

    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, k: int = 4):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n"
            "{context}\n\n"
            "Question: {question}"
        )

    def retrieve(self, query: str):
        """Return the raw retrieved documents, for eval/inspection."""
        return self.retriever.invoke(query)

    def run(self, query: str) -> str:
        """Full retrieve-then-generate call, returns the final answer string."""
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)
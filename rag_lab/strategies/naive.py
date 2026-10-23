from rag_lab.core.trace import TraceableStrategy
from langchain_core.language_models import BaseChatModel
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore


class NaiveStrategy(TraceableStrategy):
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
        """Generate from exactly the same documents captured for evaluation."""
        docs = self.retrieve(query)
        self.last_contexts = docs
        context = "\n\n".join(d.page_content for d in docs)
        return (self.prompt | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": query}
        )

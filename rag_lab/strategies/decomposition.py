"""
Decomposition retrieval strategy.

Splits a complex query into sub-questions, answers each sub-question
recursively (each answer has access to prior sub-questions' Q&A pairs
as background), and synthesizes a final answer from the full chain.

This is the most LLM-call-expensive strategy in the set: 1 decompose
call + (num_subquestions recursive-answer calls) + 1 final synthesis
call. With the default of 3 sub-questions, that's 5 calls per query —
worth batching with delay when running this against a full eval set,
same pattern used for the embedding rate-limit fix.
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from rag_lab.strategies.multi_query import LineListOutputParser


DECOMPOSITION_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant that breaks down complex questions into "
    "simpler sub-questions that can be answered independently, then "
    "combined to fully address the original question.\n\n"
    "Generate 3 sub-questions related to the input question below. "
    "Provide these sub-questions separated by newlines, with no "
    "numbering or extra commentary.\n\n"
    "Original question: {question}"
)

RECURSIVE_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Here is the question you need to answer:\n\n{sub_question}\n\n"
    "Here is any available background question + answer pairs:\n\n"
    "{qa_pairs}\n\n"
    "Here is additional context relevant to the question:\n\n"
    "{context}\n\n"
    "Use the above context and any background Q&A pairs to answer "
    "the question: {sub_question}"
)


class DecompositionStrategy:
    """Splits a complex query into sub-questions, answers each one
    recursively (building on prior sub-answers), then synthesizes
    a final answer from the full Q&A chain."""

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseChatModel,
        k: int = 4,
        num_subquestions: int = 3,
    ):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.num_subquestions = num_subquestions

        self.decompose_chain = DECOMPOSITION_PROMPT | llm | LineListOutputParser()
        self.recursive_chain = RECURSIVE_ANSWER_PROMPT | llm | StrOutputParser()

        self.final_synthesis_prompt = ChatPromptTemplate.from_template(
            "Here is a set of Q&A pairs that address different parts of "
            "a complex question:\n\n{qa_pairs}\n\n"
            "Use these to synthesize a final, complete answer to the "
            "original question: {question}"
        )

    def decompose(self, query: str) -> List[str]:
        """Split the query into sub-questions via the LLM."""
        return self.decompose_chain.invoke({"question": query})

    @staticmethod
    def _format_qa_pairs(qa_pairs: List[Tuple[str, str]]) -> str:
        return "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in qa_pairs)

    def retrieve(self, query: str) -> List[Document]:
        """Returns the union of documents retrieved across all
        sub-questions — useful for eval harness inspection, even
        though `run()` uses the recursive Q&A chain rather than this
        flat list directly."""
        sub_questions = self.decompose(query)
        all_docs: List[Document] = []
        for sub_q in sub_questions:
            all_docs.extend(self.retriever.invoke(sub_q))
        return all_docs

    def run(self, query: str) -> str:
        sub_questions = self.decompose(query)
        qa_pairs: List[Tuple[str, str]] = []

        for sub_q in sub_questions:
            docs = self.retriever.invoke(sub_q)
            context = "\n\n".join(d.page_content for d in docs)
            formatted_history = (
                self._format_qa_pairs(qa_pairs) if qa_pairs else "No prior context."
            )

            sub_answer = self.recursive_chain.invoke(
                {
                    "sub_question": sub_q,
                    "qa_pairs": formatted_history,
                    "context": context,
                }
            )
            qa_pairs.append((sub_q, sub_answer))

        final_chain = self.final_synthesis_prompt | self.llm | StrOutputParser()
        return final_chain.invoke(
            {
                "qa_pairs": self._format_qa_pairs(qa_pairs),
                "question": query,
            }
        )
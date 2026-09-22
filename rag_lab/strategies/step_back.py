"""
Step-back retrieval strategy.

Generates a more abstract, "zoomed out" version of the query, retrieves
using BOTH the original specific query and the step-back query, and
answers using the combined (but separately-labeled) context from both.

Few-shot examples in STEP_BACK_PROMPT are written in this project's own
domain (PINN/GNN/CUDA terminology) rather than generic examples, so the
model's sense of "what counts as a good abstraction" is anchored to the
corpus's actual register.

Cost: 2 LLM calls per query (step-back generation + final answer) and
2 retrieval calls (specific query + step-back query) — same shape as
HyDE's single retrieval, doubled.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.vectorstores import VectorStore
from rag_lab.utils import call_with_backoff, normalize_text

step_back_examples = [
    {
        "input": "What k-factor did Decke et al. use for their ChebConv layers?",
        "output": "What are ChebConv layers and how does the k-factor parameter affect them?",
    },
    {
        "input": (
            "Does Qian et al.'s pressure normalization scheme apply during "
            "the ghost-layer exchange or during final output?"
        ),
        "output": "How is pressure indeterminacy handled in distributed PINN training?",
    },
]

_example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt,
    examples=step_back_examples,
)

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at reframing questions. Your task is to step "
            "back and paraphrase a specific question into a more generic, "
            "conceptual question that is easier to find background context "
            "for. Here are a few examples:",
        ),
        _few_shot_prompt,
        ("human", "{question}"),
    ]
)


class StepBackStrategy:
    """Generates a more abstract version of the query, retrieves using
    both the original specific query and the step-back query, and
    answers using the combined context from both."""

    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, k: int = 4):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.step_back_chain = STEP_BACK_PROMPT | llm | StrOutputParser()

        self.answer_prompt = ChatPromptTemplate.from_template(
            "You are an expert at answering technical questions.\n\n"
            "Here is context from the specific question:\n{normal_context}\n\n"
            "Here is broader background context from a more general "
            "version of the question:\n{step_back_context}\n\n"
            "Original question: {question}\n\n"
            "Using the context above, answer the original question."
        )

    def generate_step_back_query(self, query: str) -> str:
        return normalize_text(self.step_back_chain.invoke({"question": query}))
        
    def retrieve(self, query: str) -> List[Document]:
        """Returns the union of both retrievals — specific-query docs
        first, step-back-query docs after — for eval harness inspection.
        `run()` keeps the two sources separate internally rather than
        using this flat concatenation."""
        #step_back_query = self.generate_step_back_query(query)
        #normal_docs = self.retriever.invoke(query)
        #step_back_docs = self.retriever.invoke(step_back_query)
        step_back_query = self.generate_step_back_query(query)
        normal_docs = call_with_backoff(lambda: self.retriever.invoke(query))
        step_back_docs = call_with_backoff(lambda: self.retriever.invoke(step_back_query))
        return normal_docs + step_back_docs


    def run(self, query: str) -> str:
        #step_back_query = self.generate_step_back_query(query)
        #normal_docs = self.retriever.invoke(query)
        #step_back_docs = self.retriever.invoke(step_back_query)
        step_back_query = self.generate_step_back_query(query)
        normal_docs = call_with_backoff(lambda: self.retriever.invoke(query))
        step_back_docs = call_with_backoff(lambda: self.retriever.invoke(step_back_query))
        normal_context = "\n\n".join(d.page_content for d in normal_docs)
        step_back_context = "\n\n".join(d.page_content for d in step_back_docs)
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "normal_context": normal_context,
            "step_back_context": step_back_context,
            "question": query,
        })
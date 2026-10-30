"""Optional bridge to existing RAGAS scoring notebooks.

RAGAS versions differ in accepted LLM interfaces; validate metric compatibility
on a handful of examples before launching a full benchmark.
"""
def make_ragas_llm(model=None):
    from ragas.llms import LangchainLLMWrapper
    from rag_lab.llms.gpt_adapter import build_chat_model
    return LangchainLLMWrapper(build_chat_model(model=model))

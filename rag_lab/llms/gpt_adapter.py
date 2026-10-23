"""ChatGPT Plus OAuth Responses streaming transport + LangChain adapter."""
import json, os
import requests
from rag_lab.core.cache import ResponseCache
from .chatgpt_oauth import OAuthSession, RESOURCE

class InferenceError(RuntimeError):
    def __init__(self, message, code=None, status=None):
        super().__init__(message)
        self.code, self.status = code, status

def parse_object(text):
    value = text.strip()
    if value.startswith("'''") and value.endswith("'''"):
        value = value[3:-3].strip()
    value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value

class GPTTransport:
    def __init__(self, auth=None, http=None, model=None):
        self.auth = auth or OAuthSession()
        self.http = http or requests.Session()
        self.model = model or os.environ.get("RAG_GPT_MODEL")
        if not self.model:
            raise ValueError("Set RAG_GPT_MODEL to an account-listed model slug")
        self.cache=ResponseCache(os.environ["RAG_GPT_CACHE"]) if os.environ.get("RAG_GPT_CACHE") else None
        self.calls=0
        self.cache_hits=0
        self.usage_totals={"input_tokens":0,"output_tokens":0,"total_tokens":0}

    def generate(self, input, instructions=None):
        payload = dict(model=self.model, input=input, store=False, stream=True)
        if instructions: payload["instructions"] = instructions
        key=self.cache.key(self.model,payload) if self.cache else None
        if key:
            cached=self.cache.get(key)
            if cached:
                self.cache_hits+=1
                return cached[0], {**cached[1],"cached":True}
        headers = {"Authorization": "Bearer "+self.auth.token(),
                   "Content-Type":"application/json", "Accept":"text/event-stream"}
        with self.http.post(RESOURCE+"/responses", headers=headers, json=payload,
                            stream=True, timeout=(20,240)) as r:
            if not r.ok:
                raise InferenceError("Inference failed: HTTP %d" % r.status_code, status=r.status_code)
            pieces, finished, usage = [], False, {}
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "): continue
                if line[6:] == "[DONE]": continue
                event = json.loads(line[6:])
                typ = event.get("type")
                if typ == "response.output_text.delta":
                    pieces.append(event.get("delta",""))
                elif typ == "response.completed":
                    finished = True
                    usage = (event.get("response") or {}).get("usage") or {}
                elif typ in ("response.failed","response.incomplete","error"):
                    err = (event.get("response") or {}).get("error") or event.get("error") or {}
                    code = err.get("code") if isinstance(err, dict) else None
                    raise InferenceError("Stream failed: "+str(code or typ), code=code)
            if not finished:
                raise InferenceError("Stream ended without response.completed")
            self.calls+=1
            for field in self.usage_totals:
                self.usage_totals[field]+=usage.get(field,0)
            text="".join(pieces)
            if key: self.cache.put(key,text,usage)
            return text, usage

def _convert(messages):
    instructions, items = [], []
    for m in messages:
        kind = getattr(m,"type","human")
        content = m.content if isinstance(m.content,str) else str(m.content)
        if kind in ("system","developer"): instructions.append(content)
        else: items.append({"role":"assistant" if kind=="ai" else "user", "content":content})
    if not items: items.append({"role":"user","content":"Follow the supplied instructions."})
    return items, "\n".join(instructions)

def build_chat_model(model=None, transport=None):
    """BaseChatModel for existing strategy.run() chains and structured grading."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, convert_to_messages
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import RunnableLambda
    from pydantic import PrivateAttr

    class ChatGPTPlanModel(BaseChatModel):
        model_name: str = model or os.environ.get("RAG_GPT_MODEL","")
        _transport: object = PrivateAttr()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._transport = transport or GPTTransport(model=self.model_name)

        @property
        def _llm_type(self): return "chatgpt-plan-oauth"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if stop is not None:
                raise ValueError("stop sequences not supported")
            items, instructions = _convert(messages)
            text, usage = self._transport.generate(items, instructions=instructions)
            meta = {"input_tokens":usage.get("input_tokens",0),
                    "output_tokens":usage.get("output_tokens",0),
                    "total_tokens":usage.get("total_tokens",0)}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=meta))])

        def with_structured_output(self, schema, *, include_raw=False, **kwargs):
            if kwargs: raise ValueError("Unsupported options: "+", ".join(kwargs))
            schema_json = schema.model_json_schema() if hasattr(schema,"model_json_schema") else schema
            def structured(value):
                messages = convert_to_messages(value.to_messages() if hasattr(value,"to_messages") else value)
                items, instructions = _convert(messages)
                instructions += "\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema_json)
                raw, _ = self._transport.generate(items, instructions=instructions)
                try:
                    data = parse_object(raw)
                    result = schema.model_validate(data) if hasattr(schema,"model_validate") else data
                except Exception as err:
                    if include_raw:
                        return {"raw":raw, "parsed":None, "parsing_error":err}
                    raise
                return {"raw":raw, "parsed":result, "parsing_error":None} if include_raw else result
            return RunnableLambda(structured)
    return ChatGPTPlanModel()

"""python -m rag_lab.llms.cli {login,models,test}"""
import argparse
from .chatgpt_oauth import OAuthSession
from .factory import build_transport, resolve_provider

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["login","models","test"])
    parser.add_argument("--provider", choices=("chatgpt_oauth","groq","gemini"))
    parser.add_argument("--model")
    parser.add_argument("--new-account", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    provider = resolve_provider(args.provider)
    if args.command != "test" and provider != "chatgpt_oauth":
        parser.error("login and models apply to ChatGPT OAuth; API providers use environment credentials")
    auth = OAuthSession() if provider == "chatgpt_oauth" else None
    if args.command=="login":
        result=auth.login(new_account=args.new_account, open_browser=not args.no_browser)
        print("Signed in:", result.get("email") or result["client_id"])
    elif args.command=="models":
        for m in auth.models(): print(m["slug"], ":", m["display_name"])
    else:
        answer, usage=build_transport(provider,args.model,**({'auth':auth} if auth else {})).generate([
            {"role":"user","content":"Say exactly: RAG Pipeline Lab connected."}
        ])
        print(answer, "\nToken usage:", usage)

if __name__=="__main__": main()

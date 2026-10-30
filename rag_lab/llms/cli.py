"""python -m rag_lab.llms.cli {login,models,test}"""
import argparse
from .chatgpt_oauth import OAuthSession
from .gpt_adapter import GPTTransport

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["login","models","test"])
    parser.add_argument("--new-account", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    auth = OAuthSession()
    if args.command=="login":
        result=auth.login(new_account=args.new_account, open_browser=not args.no_browser)
        print("Signed in:", result.get("email") or result["client_id"])
    elif args.command=="models":
        for m in auth.models(): print(m["slug"], ":", m["display_name"])
    else:
        answer, usage=GPTTransport(auth=auth).generate([
            {"role":"user","content":"Say exactly: RAG Pipeline Lab connected."}
        ])
        print(answer, "\nToken usage:", usage)

if __name__=="__main__": main()

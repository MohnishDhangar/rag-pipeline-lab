"""A retrieval-only CLI must NOT transitively import GPT/OAuth/JWT."""
import subprocess
import sys


def test_offline_retrieval_modules_do_not_import_oauth():
    script = '''
import builtins
original = builtins.__import__
blocked = ("rag_lab.llms", "rag_lab.evaluation.generate",
           "rag_lab.evaluation.benchmark", "rag_lab.evaluation.judge",
           "jwt")
def check(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise AssertionError("Retrieval-only module imported prohibited dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = check
from rag_lab.evaluation import retrieval_matrix, retrieval_compare
from rag_lab.retrieval.backend_factory import create_backend
from rag_lab.evaluation.jsonl_io import read_jsonl
assert retrieval_matrix.create_backend is create_backend
assert retrieval_compare.create_backend is create_backend
print("OFFLINE_RETRIEVAL_IMPORTS_OK")
'''
    completed=subprocess.run(
        [sys.executable,"-c",script],text=True,capture_output=True,check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OFFLINE_RETRIEVAL_IMPORTS_OK" in completed.stdout

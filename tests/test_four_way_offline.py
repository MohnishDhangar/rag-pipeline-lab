"""Four-way reporting must not load GPT OAuth or require PyJWT."""
import subprocess
import sys


def test_four_way_report_imports_with_oauth_blocked():
    script="""
import builtins
original=builtins.__import__
blocked=('rag_lab.llms','rag_lab.evaluation.judge','rag_lab.evaluation.benchmark','jwt')
def guarded(name,*args,**kwargs):
    if any(name == p or name.startswith(p+'.') for p in blocked):
        raise AssertionError('Forbidden import in offline report: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from rag_lab.evaluation.four_way_report import report
from rag_lab.evaluation.identity import row_fingerprint, RUBRIC_VERSION
print(RUBRIC_VERSION)
"""
    out=subprocess.run([sys.executable,"-c",script],capture_output=True,
                       text=True,check=False)
    assert out.returncode==0,out.stderr
    assert "scientific-judge-v1" in out.stdout

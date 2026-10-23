"""Train a router on held-out question IDs, not on unscored generation results.

Example:
 python -m rag_lab.routing.train_router --generations eval/generated_gpt.jsonl \
    --scores eval/gpt_judge_scores.jsonl --output data/adaptive_router.joblib
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from .query_features import features, FEATURE_NAMES
from rag_lab.evaluation.benchmark import read_jsonl

DEFAULT_CALLS={"naive":1,"multi_query":2,"rag_fusion":2,"hyde":2,
               "step_back":2,"decomposition":5,"crag":8}

def prepare_rows(generations, scores, *, lambda_calls=0.025, quality_key="answer_correctness"):
    score_map={(r.get("strategy"),r.get("backend"),r.get("id")):r
               for r in scores if isinstance(r.get(quality_key),(int,float))}
    grouped=defaultdict(list)
    for row in generations:
        score=score_map.get((row.get("strategy"),row.get("backend"),row.get("id")))
        if not score or not row.get("answer"): continue
        name=row["strategy"]
        utility=float(score[quality_key])-lambda_calls*DEFAULT_CALLS.get(name,3)
        grouped[row["id"]].append((utility,name,row["question"]))
    labeled=[]
    for q_id, options in grouped.items():
        strategies={name for _,name,_ in options}
        if len(strategies)<2: continue  # cannot establish a meaningful choice
        best=max(options,key=lambda x:x[0])
        labeled.append((q_id,best[2],best[1],best[0]))
    return labeled

def train(generations,scores,output,seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    examples=prepare_rows(generations,scores)
    if len(examples)<25:
        raise ValueError("Need at least 25 distinct questions with two or more scored strategies")
    questions=[q for _,q,_,_ in examples]
    labels=[label for _,_,label,_ in examples]
    if len(set(labels))<2:
        raise ValueError("Training labels have only one winning strategy")
    x=[features(q) for q in questions]
    indices=list(range(len(examples)))
    train_i,test_i=train_test_split(indices,test_size=0.25,random_state=seed)
    model=RandomForestClassifier(n_estimators=120,max_depth=5,
                                 min_samples_leaf=2,random_state=seed,
                                 class_weight="balanced")
    model.fit([x[i] for i in train_i],[labels[i] for i in train_i])
    accuracy=sum(model.predict([x[i]])[0]==labels[i] for i in test_i)/len(test_i)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"features":FEATURE_NAMES,"model":model,
                 "holdout_accuracy":accuracy,"train_questions":len(train_i),
                 "test_questions":len(test_i)},output)
    return {"train_questions":len(train_i),"heldout_questions":len(test_i),
            "heldout_top_strategy_accuracy":accuracy,
            "label_counts":{v:labels.count(v) for v in sorted(set(labels))}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generations",default="eval/generated_gpt.jsonl")
    p.add_argument("--scores",default="eval/gpt_judge_scores.jsonl")
    p.add_argument("--output",default="data/adaptive_router.joblib")
    args=p.parse_args()
    report=train(read_jsonl(args.generations),read_jsonl(args.scores),args.output)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()

import json
import pandas as pd
from tqdm import tqdm
import re

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)               
    text = re.sub(r'\(see.*?\)', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_jsonl(path):
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # handle different possible field names
            if "judgment_text" in obj:
                text = obj["judgment_text"]
            elif "text" in obj:
                text = obj["text"]
            elif "body" in obj:
                text = obj["body"]
            else:
                # fallback combine anything text-like
                text = " ".join([str(v) for k, v in obj.items() if isinstance(v, str)])

            doc_id = obj.get("doc_id", obj.get("case_id", obj.get("id", None)))

            documents.append({
                "case_id": doc_id,
                "judgment_text": text
            })
    return pd.DataFrame(documents)

def process_dataset():
    train = load_jsonl("data/raw/train.jsonl")
    dev   = load_jsonl("data/raw/dev.jsonl")
    test  = load_jsonl("data/raw/test.jsonl")

    df = pd.concat([train, dev, test], ignore_index=True)

    tqdm.pandas()
    df["clean_text"] = df["judgment_text"].progress_apply(clean_text)

    df.to_csv("data/processed/clean_cases.csv", index=False)
    print("Saved → data/processed/clean_cases.csv")
    print(df.head())

if __name__ == "__main__":
    process_dataset()

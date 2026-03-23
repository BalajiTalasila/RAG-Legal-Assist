import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

index = faiss.read_index("data/processed/legal.index")
chunks = pd.read_pickle("data/processed/chunks.pkl")

embedder = SentenceTransformer("BAAI/bge-large-en")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


def retrieve(query, top_k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)

    retrieved_chunks = []
    for idx in I[0]:
        retrieved_chunks.append(chunks.iloc[idx]["chunk"])

    return retrieved_chunks

def generate_answer(query):
    retrieved = retrieve(query)

    context_block = "\n\n---\n\n".join(retrieved)

    prompt = f"""
You are a legal assistant.
Use ONLY the provided legal context to answer.
Always cite chunk references like (Chunk #).

Query:
{query}

Context:
{context_block}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_length=512,
        do_sample=True,
        temperature=0.2
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    q = "What legal grounds were used in this case?"
    print(generate_answer(q))

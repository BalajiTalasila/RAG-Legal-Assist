import faiss
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ahlc_chunker import adaptive_hybrid_chunk
import os

model = SentenceTransformer("BAAI/bge-large-en")

def build_index(input_csv, index_path):
    df = pd.read_csv(input_csv)

    texts = []
    meta_case_id = []
    meta_chunk_id = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = str(row["case_id"])
        text = str(row["clean_text"])

        chunks = adaptive_hybrid_chunk(text)

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            meta_case_id.append(case_id)
            meta_chunk_id.append(i)

    print(f"Total chunks created: {len(texts)}")

    embeddings = model.encode(texts, batch_size=16, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    faiss.write_index(index, index_path)

    meta_df = pd.DataFrame({
        "chunk": texts,
        "case_id": meta_case_id,
        "chunk_id": meta_chunk_id
    })

    meta_df.to_pickle("data/processed/chunks.pkl")

    print("FAISS Index + Chunks stored successfully")

if __name__ == "__main__":
    build_index("data/processed/clean_cases.csv", "data/processed/legal.index")

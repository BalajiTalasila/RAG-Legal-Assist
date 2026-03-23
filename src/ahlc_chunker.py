import re
import nltk
from sentence_transformers import SentenceTransformer, util

# Download only once (handles Windows nicely)
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SECTION_PATTERNS = [
    r"Section\s+\d+",
    r"Article\s+\d+",
    r"Part\s+\d+",
    r"Chapter\s+\d+",
    r"§\s*\d+",
]

def rule_based_split(text):
    pattern = "(" + "|".join(SECTION_PATTERNS) + ")"
    parts = re.split(pattern, text)

    chunks = []
    for i in range(1, len(parts), 2):
        section_title = parts[i]
        content = parts[i + 1]
        chunks.append(section_title + " " + content)

    return chunks if chunks else [text]

def semantic_split(text, threshold=0.55):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 2:
        return [text]

    embeddings = MODEL.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i])

        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

        current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))
    return chunks

def enforce_length(chunks, max_tokens=450):
    final_chunks = []

    for ch in chunks:
        words = ch.split()

        while len(words) > max_tokens:
            final_chunks.append(" ".join(words[:max_tokens]))
            words = words[max_tokens:]

        if words:
            final_chunks.append(" ".join(words))

    return final_chunks

def adaptive_hybrid_chunk(text):
    # Step 1: Try structure first
    rb_chunks = rule_based_split(text)

    # If structural segmentation exists → use it
    if len(rb_chunks) > 1:
        chunks = rb_chunks
    else:
        # Otherwise → semantic segmentation
        chunks = semantic_split(text)

    # Final → enforce safe LLM size
    chunks = enforce_length(chunks)

    return chunks

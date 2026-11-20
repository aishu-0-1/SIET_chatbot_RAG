
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. Load PDF
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError("No readable text found in PDF")

    return full_text


# 2. Create chunks
def create_chunks(full_text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# 3. Build vectors
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot create embeddings without chunks")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(embed_vectors, dtype="float32")

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return index, embed_model


# 4. Load base
def load_base(pdf_path="01_College_Information.pdf"):
    full_text = load_pdf(pdf_path)
    chunks = create_chunks(full_text)

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# 5. Retrieve content
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(query_embed, dtype="float32")

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i >= 0:
            background += chunks[i] + "\n\n"

    return background


# 6. Ask RAG
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):
    if not api_key:
        raise ValueError("API Key Missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    prompt = f"""
You are a professional AI assistant for Shridevi Institute of Engineering and Technology.

Answer the user's question using ONLY the provided document background.

Rules:
- Give correct and clear answers.
- Do not invent facts.
- Do not use outside knowledge.
- If the answer is missing from the document, say "Information Missing".
- Keep the answer simple and relevant.

Document Background:
{background}

User Question:
{question}

Answer:
"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

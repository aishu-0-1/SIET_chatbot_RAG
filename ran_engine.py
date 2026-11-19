import faiss
import numpy as np
from groq import Groq
from pypdf import PDFReader
from sentence_tranformers import SentenceTransformer


#load pdfs
def load_pdf(pdd_path):
  reader = PDFReader(str(pdf_path))
  mypages = [i.extract_text() or "" for i in reader.pages]
  full_text ="\n".join(mypages).strip()
  if not  full_text:
    raise ValueError("No readable Text found")
    return full_text

#chunking
def create_chunks(full_text, chunksize =500, overlap =100):
  chunks = []
  start = 0
  while start<len(full_text):        #0/1000                   400<1000              800<1000
    end = start + chunksize          #500                     400 +500=900           800+500=1300
    chunk = full_text[start:end]     #[0:500]                 [400:900]              [800:1300]
    if chunk.strip():
      chunks.append(chunk)
    start = end - overlap             #500-100=400         900-100 =800             1300-100=1200
   return chunks

#Embedding:
def build_vectors(chunks, model_name ="all-MiniLM-L6-v2"):
  if not chunk:
    raise ValueError("Cannot apply embedding without chunks")
embed_model = SentenceTransforemer(model_name)
embed_vectors = embed_model.encode(chunks, covert_to_numpy = True)
faiss.normalize_L2(embed_vectors)
index = faiss.IndexFlatIP(embed_vectors.shape[1])
index.add(embed_vectors)
return index, embed_model

#Call loadpdf, chunk, embedding
def load_base(pdf_path = "01_College_Information.pdf"):
  full_text = load_pdf(pdf_path)
  chunk = create_chunks(full_text)
embed_model, index = build_vector(chunks)
return embed_model, chunks ,index

#retrieve
def retreve_content (question, embed_model, chunks, index, num_chunks=3):
   if not question.strip():
    raise ValueError("Question cannot be blank")
    if num_ckunks < 1:
      raise ValueError("Request atleast one or more chunks")
quert_embed = embed_model.encode([question], covert_to_numpy=True)
faiss.normalize_L2(query_embed)
sim_scores, vec_indices = index.search(query_embed,min(num_chunks, len(chunks)))
background = ""
for i in vec_indices[0]:
  background += chunks[i] + "\n\n"
return background

#Ask:
def ask_rag (question, embed_model, chunks, index, api_key, num_chunks, model_name = "openai/gpt-oss-120b"):
  key = api_key
  if not key:
    raise ValueError("API Key Missing")
  background = retrive_content(question, embed_model, chunks, index, num_chunk=3)
  prompt = """ You are an professional AI Assitant.
  Do not invent or assume facts.
  If information not available, simply say Information Missing.
  Document Background :""" + background + "user Question" + question + ""Answer:"
  client = Groq(api_key =key)
  response = client.chat.completions.create(
        model = model_name,
        messages = [{"role":"user",
                     "content": prompt}],
                      temperature = 0.3,
  )
return response.choices[0].message.content


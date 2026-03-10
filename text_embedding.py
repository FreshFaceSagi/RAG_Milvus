from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class Embedding:

    def do_embedding(self, texts):
      return model.encode(texts)
    
    def get_chunks(self, text, chunk_size=400, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Move start forward, but stay back by 'overlap' amount
            start += (chunk_size - overlap)
        return chunks
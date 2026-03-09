from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class Embedding:

    def do_embedding(self, texts):
      return model.encode(texts)

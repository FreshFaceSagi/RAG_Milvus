from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from text_embedding import Embedding
from pymilvus import MilvusClient as Client

class MilvusClient:
    
    def __init__(self):
        try:
            self.client = Client("http://localhost:19530","root","milvus")
            print(f" self.client :{self.client }")
        except Exception as e:
          print(e)
    #   connections.connect(
    #    host="localhost",
    #     port="19530"
    #    )

    def creation_collection(self):
        fields = [
           FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
           FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
           FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
              ]

        schema = CollectionSchema(fields, description="example collection")

        self.collection = Collection(
         name="documents_collection",
           schema=schema
            )

        print("Collection created")
  
    def search(self, collection_name, query_vector):

      print("Initiated search")

      search_params = {
        "metric_type": "COSINE",   # or L2 depending on your index
        "params": {"nprobe": 10}
      }
      self.collection.load()
      results = self.client.search(
        collection_name=collection_name,
        data=[query_vector],          # must be list of vectors
        anns_field="embedding",
        search_params=search_params,
        limit=3,
        output_fields=["text"]
        )

      return results


    def insert(self, data):
       print("Inserting into VectorDB")
       self.collection.insert(data)

print("Connecting to the milvus Milvus!")
client = MilvusClient()
#client.get_connection()
embedding = Embedding()
# print(connections.has_connection("default"))
# print("creating collections")
# client.creation_collection()

# texts = [
#     "Milvus is a vector database",
#     "Artificial intelligence is transforming technology",
#     "Python is a popular programming language"
# ]

# embedding = Embedding()
# vectors = embedding.do_embedding(texts)
# data = [
#     texts,
#     vectors.tolist()
# ]
# print(vectors.tolist())
# client.insert(data)


query = "What is a vector database?"

query_vector = embedding.do_embedding(query).tolist()
print(query_vector)
print(client.search("documents_collection",query_vector))
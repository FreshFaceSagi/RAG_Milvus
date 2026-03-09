from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from text_embedding import Embedding
from pymilvus import MilvusClient as Client

class MilvusClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        # Initialize connections and collections here

    def creation_collection(self, collection_name):
        # Logic to create a collection
        pass
    
    def __init__(self):
        print("Initializing Milvus Client")
        try:
            self.collection = None
            self.client = Client("http://localhost:19530", "root", "milvus")
            print(f"Client connected: {self.client}")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")

    def creation_collection(self):
        print("Creating collection")
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

        print("Collection created successfully")
  
    def search(self, collection_name, query_vector):
        print("Initiated search")
        
        if not self.collection:
            print("Collection not found and creating the collection")
            self.creation_collection()
        print(f"Using collection: {self.collection}")
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        try:
            self.collection.load()
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="embedding",
                search_params=search_params,
                limit=3,
                output_fields=["text"]
            )
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return None

    def insert(self, data):
        # Logic to insert data into the collection
        pass

    def search(self, query):
        # Logic to search in the collection
        pass

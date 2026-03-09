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

    def insert(self, data):
        # Logic to insert data into the collection
        pass

    def search(self, query):
        # Logic to search in the collection
        pass

import milvus
def insert_data(collection_name, data):
    # Initialize Milvus client
    client = milvus.Milvus(uri='tcp://localhost:19530')
    
    # Check if collection exists
    if not client.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return
    
    # Insert data into the specified collection
    status, ids = client.insert(collection_name, data)
    if status.code == milvus.StatusCode.SUCCESS:
        print(f"Inserted {len(data)} records into '{collection_name}' with IDs: {ids}")
    else:
        print(f"Failed to insert data: {status.message}")

# Sample usage
if __name__ == '__main__':
    sample_collection = 'example_collection'
    sample_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]  # Replace with your actual data
    insert_data(sample_collection, sample_data)
# This file handles logic to connect and store in ChromaDB
import chromadb
from chromadb.config import Settings

class VectorDBService:
    def __init__(self):
        # Initialize the ChromaDB client
        # 'path' indicates where the physical data will be saved in your folder
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or retrieve the collection named "llm_cache"
        # We will use 'cosine' distance (cosine similarity) which is ideal for text
        self.collection = self.client.get_or_create_collection(
            name="llm_cache",
            metadata={"hnsw:space": "cosine"} 
        )

    def add_to_cache(self, vector, response, prompt, doc_id):
        """
        Saves the question vector and the LLM response.
        """
        self.collection.add(
            embeddings=[vector],      # The numeric vector
            documents=[response],    # The text response we want to retrieve
            metadatas=[{"prompt": prompt}], # Extra info (optional)
            ids=[doc_id]             # A unique ID
        )

    def search_cache(self, vector, n_results=1):
        """
        Searches for the closest vector in the database.
        """
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results
        )
        return results

# Global instance to use throughout the app
vector_db = VectorDBService()
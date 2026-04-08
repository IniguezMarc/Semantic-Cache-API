from sentence_transformers import SentenceTransformer
import requests
import os

class LocalLLMService:
    def __init__(self):
        print("Loading local embeddings model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        
        # Read the Ollama URL from environment variables (Docker will configure this)
        # If it doesn't exist, we use localhost by default
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def get_embedding(self, text: str) -> list[float]:
        vector = self.embedding_model.encode(text).tolist()
        return vector

    def generate_real_response(self, prompt: str) -> str:
        print(f"--> [OLLAMA] Requesting response for: '{prompt}'")
        
        # Make the request to the Ollama API
        # We will use the 'llama3' or 'mistral' model, whichever you prefer to download
        payload = {
            "model": "llama3", 
            "prompt": prompt,
            "stream": False # We want the complete response, not word by word
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status() # Throws an error if the connection fails
            data = response.json()
            return data.get("response", "Error reading the response from Ollama.")
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return "Sorry, the local LLM model is not available at this moment."

# Global instance
llm_service = LocalLLMService()

from fastapi import FastAPI
from pydantic import BaseModel
import hashlib

# Import our local services
from services.vector_db import vector_db
from services.llm_service import llm_service

app = FastAPI(title="Semantic Cache API")

class UserRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_question(request: UserRequest):
    prompt = request.prompt
    print(f"\n--- New request received: '{prompt}' ---")

    # 1. Convert the question to a vector (our 384 numbers)
    print("1. Generating local embedding...")
    vector = llm_service.get_embedding(prompt)
    
    # 2. Search in ChromaDB
    print("2. Searching in the vector memory...")
    search_results = vector_db.search_cache(vector, n_results=1)
    
    # 3. Analyze results and the DISTANCE THRESHOLD
    # In ChromaDB Cosine distance: 0.0 is identical, 1.0 is completely different.
    THRESHOLD = 0.4 
    
    if search_results['distances'] and search_results['distances'][0]:
        distance = search_results['distances'][0][0]
        print(f"   -> Found candidate with distance: {distance:.4f}")
        
        if distance < THRESHOLD:
            # CACHE HIT! Very similar meaning.
            print("   -> BINGO! The distance is below the threshold. Using cache.")
            saved_response = search_results['documents'][0][0]
            original_prompt = search_results['metadatas'][0][0]['prompt']
            
            return {
                "status": "cache_hit",
                "distance": round(distance, 4),
                "original_prompt_matched": original_prompt,
                "response": saved_response
            }
        else:
            print("   -> The distance is above the threshold. They are not similar enough.")
    else:
         print("   -> The cache is empty.")

    # 4. CACHE MISS! No good matches found, we must "pay" to generate the response
    print("3. Generating new response (Ollama)...")
    new_response = llm_service.generate_real_response(prompt)
    
    # --- NEW SAFETY NET ---
    # We only save to cache if the response is NOT our default error message
    if "Sorry, the local LLM model is not available" not in new_response:
        print("4. Saving into the vector database for the future...")
        # Create a unique and deterministic ID using the prompt text
        doc_id = hashlib.md5(prompt.encode('utf-8')).hexdigest() 
        
        vector_db.add_to_cache(
            vector=vector, 
            response=new_response, 
            prompt=prompt, 
            doc_id=doc_id
        )
    else:
        print("4. Error detected. Not saving to cache to prevent poisoning.")
        
    return {
        "status": "cache_miss (new generation)",
        "distance": None,
        "response": new_response
    }
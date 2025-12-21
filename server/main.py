from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
import sys

sys.path.append(os.path.abspath("build"))
import veloxdb

class VectorPayload(BaseModel):
    vector: list[float]

class SearchPayload(BaseModel):
    query_vector: list[float]
    metric: str = "eucl"

class TrainPayload(BaseModel):
    num_clusters: int
    max_iters: int
    metric: str = "eucl"

DATA_FILE = "data/vectors.fvecs"
INDEX_FILE = "data/index.ivf"

db = veloxdb.VectorIndex()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("Serve is starting up..")
    if not os.path.exists("data"):
        os.makedirs("data")

    if os.path.exists(DATA_FILE) and os.path.exists(INDEX_FILE):
        print("Loading existing dataset and index...")
        try:
            db.load_fvecs(DATA_FILE)
            db.load_index(INDEX_FILE)
            print("Dataset and index loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset or index: {e}")
    else:
        print("No saved state found. Create a new data store to get started!")

@app.get("/")
def health_check():
    return {"status": "running", "message": "Server is operational."}

@app.post("/add_vectors")
def add_vectors(payload: VectorPayload):
    """Add a new vector to the database."""
    try:
        db.add_vector(payload.vector)
        return {"status":"success", "message": "Vector added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/train")
def train_index(payload: TrainPayload):
    """Build or train the index with specified parameters."""
    try:
        print(f"Training index with parameters: {payload.num_clusters} clusters, {payload.max_iters} max_iters, metric: {payload.metric}")
        db.build_index(num_clusters=payload.num_clusters, max_iters=payload.max_iters, metric=payload.metric)
        return {"status":"success", "message": "Index trained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(payload: SearchPayload):
    """Search for nearest neighbors of the query vector."""
    try:
        result_id = db.search(payload.query_vector, metric=payload.metric)
        return {"status" : "success", "match_id" : result_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/save")
def save_state():
    """Save the current state of the database and index to disk."""
    try:
        db.write_fvecs(DATA_FILE)
        db.save_index(INDEX_FILE)
        return {"status":"success", "message": "State saved successfully.", "files": [DATA_FILE, INDEX_FILE]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
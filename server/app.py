from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from server import embedder, state
from server.fvecs_io import read_fvecs
from server.metadata import MetadataStore
from server.schemas import (
    BatchDocumentsPayload,
    DocumentPayload,
    EmbedPayload,
    SearchPayload,
    TrainPayload,
    VectorPayload,
)

metadata = MetadataStore(state.METADATA_FILE)

app = FastAPI(title="VeloxDB API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_dim(expected_len: int) -> None:
    if state.dim is not None and state.dim != expected_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Vector dimension mismatch: database dim={state.dim}, "
                f"got {expected_len}. Clear data/ or use matching embeddings."
            ),
        )


def _add_vector_to_db(vector: list[float]) -> int:
    _check_dim(len(vector))
    doc_id = state.vector_count
    state.db.add_vector(vector)
    state.vector_count += 1
    if state.dim is None:
        state.dim = len(vector)
    state.is_indexed = False
    state.refresh_stats()
    return doc_id


@app.on_event("startup")
async def startup_event():
    print("Server is starting up...")
    state.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if state.DATA_FILE.exists():
        print("Hydrating vectors from disk into RAM...")
        try:
            vectors = read_fvecs(state.DATA_FILE)
            for vec in vectors:
                state.db.add_vector(vec)
            state.vector_count = len(vectors)
            state.refresh_stats()
            print(f"Loaded {state.vector_count} vectors (dim={state.dim}).")
        except Exception as e:
            print(f"Error hydrating vectors: {e}")

    if state.INDEX_FILE.exists() and state.vector_count > 0:
        try:
            state.db.load_index(str(state.INDEX_FILE))
            state.is_indexed = True
            print("IVF index loaded successfully.")
        except Exception as e:
            print(f"Error loading index: {e}")
            state.is_indexed = False

    metadata.load()
    if metadata.count() != state.vector_count:
        print(
            f"Warning: metadata count ({metadata.count()}) != "
            f"vector count ({state.vector_count})"
        )
    elif not state.DATA_FILE.exists():
        print("No saved state found. Create documents to get started!")


def _health_payload() -> dict:
    return {
        "status": "running",
        "message": "Server is operational.",
        "vector_count": state.vector_count,
        "dim": state.dim,
        "is_indexed": state.is_indexed,
        "expected_embedding_dim": embedder.embedding_dim(),
    }


@app.get("/")
def root():
    return _health_payload()


@app.get("/health")
def health():
    return _health_payload()


@app.post("/embed")
def embed_text(payload: EmbedPayload):
    try:
        vector = embedder.embed(payload.text)
        return {"status": "success", "vector": vector, "dim": len(vector)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents")
def add_document(payload: DocumentPayload):
    try:
        vector = embedder.embed(payload.text)
        doc_id = _add_vector_to_db(vector)
        metadata.add(doc_id, payload.text)
        return {"status": "success", "id": doc_id, "dim": len(vector)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents/batch")
def add_documents_batch(payload: BatchDocumentsPayload):
    try:
        ids = []
        for text in payload.texts:
            stripped = text.strip()
            if not stripped:
                continue
            vector = embedder.embed(stripped)
            doc_id = _add_vector_to_db(vector)
            metadata.add(doc_id, stripped)
            ids.append(doc_id)
        return {"status": "success", "ids": ids, "count": len(ids)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/documents")
def list_documents(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    docs = metadata.list_all()
    page = docs[offset : offset + limit]
    return {
        "status": "success",
        "total": len(docs),
        "offset": offset,
        "limit": limit,
        "documents": page,
    }


@app.get("/documents/{doc_id}")
def get_document(doc_id: int, preview_dims: int = Query(8, ge=0, le=64)):
    if doc_id < 0 or doc_id >= state.vector_count:
        raise HTTPException(status_code=404, detail="Document not found")
    meta = metadata.get(doc_id)
    try:
        vector = state.db.get_vector(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    preview = vector[:preview_dims] if preview_dims else []
    return {
        "status": "success",
        "id": doc_id,
        "text": meta["text"] if meta else None,
        "created_at": meta.get("created_at") if meta else None,
        "vector_preview": preview,
        "dim": len(vector),
    }


@app.post("/add_vectors")
def add_vectors(payload: VectorPayload):
    try:
        doc_id = _add_vector_to_db(payload.vector)
        return {
            "status": "success",
            "message": "Vector added successfully.",
            "id": doc_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/train")
def train_index(payload: TrainPayload):
    if state.vector_count == 0:
        raise HTTPException(status_code=400, detail="No vectors to train on")
    try:
        state.db.build_index(
            num_clusters=payload.num_clusters,
            max_iters=payload.max_iters,
            metric=payload.metric,
        )
        state.is_indexed = True
        return {"status": "success", "message": "Index trained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/search")
def search(payload: SearchPayload):
    if state.vector_count == 0:
        raise HTTPException(status_code=400, detail="No vectors in database")
    try:
        if payload.query_text is not None:
            query_vector = embedder.embed(payload.query_text.strip())
        else:
            query_vector = payload.query_vector
        _check_dim(len(query_vector))
        match_id = state.db.search(query_vector, metric=payload.metric)
        meta = metadata.get(match_id)
        result = {
            "id": match_id,
            "text": meta["text"] if meta else None,
        }
        return {"status": "success", "results": [result], "is_indexed": state.is_indexed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/save")
def save_state():
    if state.vector_count == 0:
        raise HTTPException(status_code=400, detail="No vectors to save")
    try:
        state.db.write_fvecs(str(state.DATA_FILE))
        if state.is_indexed:
            state.db.save_index(str(state.INDEX_FILE))
        metadata.save()
        files = [str(state.DATA_FILE), str(state.METADATA_FILE)]
        if state.is_indexed:
            files.append(str(state.INDEX_FILE))
        return {
            "status": "success",
            "message": "State saved successfully.",
            "files": files,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

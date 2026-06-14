from pydantic import BaseModel, Field, model_validator


class VectorPayload(BaseModel):
    vector: list[float]


class TrainPayload(BaseModel):
    num_clusters: int
    epochs: int = 10
    metric: str = "eucl"


class SearchPayload(BaseModel):
    query_vector: list[float] | None = None
    query_text: str | None = None
    metric: str = "eucl"
    k: int = Field(default=5, ge=1, le=100)
    nprobe: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def require_query(self):
        has_vector = self.query_vector is not None and len(self.query_vector) > 0
        has_text = self.query_text is not None and self.query_text.strip() != ""
        if has_vector == has_text:
            raise ValueError("Provide exactly one of query_vector or query_text")
        return self


class DocumentPayload(BaseModel):
    text: str = Field(min_length=1)


class BatchDocumentsPayload(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class EmbedPayload(BaseModel):
    text: str = Field(min_length=1)

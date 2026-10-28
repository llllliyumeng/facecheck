from fastapi import FastAPI

app = FastAPI(
    title="Facecheck",
    description="Privacy-preserving portrait video storage prototype",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}


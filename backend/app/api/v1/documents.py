import json
import uuid
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory store for document metadata
_documents: dict[str, dict] = {}
_doc_service: DocumentService | None = None


def _get_doc_service() -> DocumentService:
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService()
    return _doc_service


@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(...),
    metadata: str = Form("{}"),
):
    """Upload and ingest a document into the vector store."""
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    try:
        result = await _get_doc_service().ingest(
            file=file,
            collection=collection,
            metadata=meta,
        )
        _documents[result["id"]] = result
        return result
    except Exception as e:
        doc_id = str(uuid.uuid4())
        error_record = {
            "id": doc_id,
            "filename": file.filename,
            "collection": collection,
            "chunk_count": 0,
            "status": "error",
            "error": str(e),
        }
        _documents[doc_id] = error_record
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_documents(collection: str | None = None):
    """List documents, optionally filtered by collection."""
    docs = list(_documents.values())
    if collection:
        docs = [d for d in docs if d.get("collection") == collection]
    return docs


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get document details."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the vector store and metadata."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        await _get_doc_service().delete(doc["collection"], document_id)
    except Exception:
        pass  # Best effort deletion from vector store

    del _documents[document_id]
    return {"status": "deleted"}

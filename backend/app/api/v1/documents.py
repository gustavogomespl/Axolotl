import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.document import Document
from app.models.project import Project
from app.services.document_service import DocumentService

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])

_doc_service: DocumentService | None = None


def _get_doc_service() -> DocumentService:
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService()
    return _doc_service


def _doc_to_response(doc: Document) -> dict:
    return {
        "id": doc.id,
        "project_id": doc.project_id,
        "filename": doc.filename,
        "collection_name": doc.collection_name,
        "content_type": doc.content_type,
        "chunk_count": doc.chunk_count,
        "status": doc.status,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat() if doc.created_at else "",
    }


@router.post("")
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    collection: str = Form(...),
    metadata: str = Form("{}"),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    doc = Document(
        project_id=project_id,
        filename=file.filename or "unknown",
        collection_name=collection,
        content_type=file.content_type or "application/octet-stream",
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        result = await _get_doc_service().ingest(file=file, collection=collection, metadata=meta)
        doc.chunk_count = result.get("chunk_count", 0)
        doc.status = "ready"
    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)

    await db.commit()
    await db.refresh(doc)
    return _doc_to_response(doc)


@router.get("")
async def list_documents(
    project_id: str, collection: str | None = None, db: AsyncSession = Depends(get_db)
):
    query = select(Document).where(Document.project_id == project_id)
    if collection:
        query = query.where(Document.collection_name == collection)
    query = query.order_by(Document.created_at.desc())
    result = await db.execute(query)
    return [_doc_to_response(d) for d in result.scalars().all()]


@router.get("/{document_id}")
async def get_document(project_id: str, document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.project_id == project_id)
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_response(doc)


@router.delete("/{document_id}")
async def delete_document(project_id: str, document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.project_id == project_id)
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        await _get_doc_service().delete(doc.collection_name, document_id)
    except Exception:
        pass  # Best effort deletion from vector store

    await db.delete(doc)
    await db.commit()
    return {"status": "deleted"}

"""Tests for app.services.document_service.DocumentService."""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.add_documents = MagicMock(return_value=["id-1", "id-2"])
    vs.get_or_create_collection = MagicMock()
    return vs


@pytest.fixture
def service(mock_vector_store):
    from app.services.document_service import DocumentService

    return DocumentService(vector_store=mock_vector_store)


def _make_upload_file(filename: str, content: bytes = b"hello world") -> MagicMock:
    """Create a mock UploadFile with async read()."""
    upload = MagicMock()
    upload.filename = filename
    upload.read = AsyncMock(return_value=content)
    return upload


# ---------------------------------------------------------------------------
# ingest() -- PDF
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_pdf_success(service, mock_vector_store):
    """Successful PDF upload: save -> parse -> chunk -> embed -> cleanup."""
    upload = _make_upload_file("report.pdf", b"%PDF-fake-content")

    with (
        patch("app.services.document_service.tempfile") as mock_tempfile,
        patch("app.services.document_service.os") as mock_os,
        patch("app.services.document_service.RecursiveCharacterTextSplitter") as MockSplitter,
        patch("app.services.document_service.uuid") as mock_uuid,
    ):
        # Setup tempfile
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/fake123.pdf"
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        mock_os.path.splitext.return_value = ("report", ".pdf")

        # Setup splitter
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["chunk1", "chunk2", "chunk3"]
        MockSplitter.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value = "test-doc-id"

        # Mock _parse_file to avoid actually parsing a PDF
        with patch.object(service, "_parse_file", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "Parsed PDF text content"

            result = await service.ingest(
                file=upload,
                collection="my-collection",
                chunk_size=500,
                chunk_overlap=100,
            )

    # Verify file was read
    upload.read.assert_awaited_once()

    # Verify temp file was written
    mock_tmp.write.assert_called_once_with(b"%PDF-fake-content")

    # Verify parse was called
    mock_parse.assert_awaited_once_with("/tmp/fake123.pdf", ".pdf")

    # Verify splitter configuration
    MockSplitter.assert_called_once_with(chunk_size=500, chunk_overlap=100)
    mock_splitter_instance.split_text.assert_called_once_with("Parsed PDF text content")

    # Verify documents added to vector store
    mock_vector_store.add_documents.assert_called_once()
    call_args = mock_vector_store.add_documents.call_args
    assert call_args[0][0] == "my-collection"
    docs = call_args[0][1]
    assert len(docs) == 3
    assert docs[0].page_content == "chunk1"
    assert docs[0].metadata["document_id"] == "test-doc-id"
    assert docs[0].metadata["filename"] == "report.pdf"
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[0].metadata["total_chunks"] == 3

    # Verify cleanup
    mock_os.unlink.assert_called_once_with("/tmp/fake123.pdf")

    # Verify return structure
    assert result["id"] == "test-doc-id"
    assert result["filename"] == "report.pdf"
    assert result["collection"] == "my-collection"
    assert result["chunk_count"] == 3
    assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# ingest() -- TXT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_txt_success(service, mock_vector_store):
    """Successful TXT upload goes through the same pipeline."""
    upload = _make_upload_file("notes.txt", b"Some plain text notes.")

    with (
        patch("app.services.document_service.tempfile") as mock_tempfile,
        patch("app.services.document_service.os") as mock_os,
        patch("app.services.document_service.RecursiveCharacterTextSplitter") as MockSplitter,
        patch("app.services.document_service.uuid") as mock_uuid,
    ):
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/fake456.txt"
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp
        mock_os.path.splitext.return_value = ("notes", ".txt")

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["chunk-a"]
        MockSplitter.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value = "txt-doc-id"

        with patch.object(service, "_parse_file", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "Some plain text notes."

            result = await service.ingest(
                file=upload,
                collection="text-coll",
            )

    assert result["filename"] == "notes.txt"
    assert result["chunk_count"] == 1
    assert result["status"] == "ready"

    # Default chunk_size / chunk_overlap used
    MockSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)


# ---------------------------------------------------------------------------
# ingest() -- with metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_with_metadata(service, mock_vector_store):
    """Custom metadata dict is merged into each chunk's metadata."""
    upload = _make_upload_file("data.md", b"# Heading\nContent")

    with (
        patch("app.services.document_service.tempfile") as mock_tempfile,
        patch("app.services.document_service.os") as mock_os,
        patch("app.services.document_service.RecursiveCharacterTextSplitter") as MockSplitter,
        patch("app.services.document_service.uuid") as mock_uuid,
    ):
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/meta.md"
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp
        mock_os.path.splitext.return_value = ("data", ".md")

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["c1"]
        MockSplitter.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value = "meta-id"

        with patch.object(service, "_parse_file", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "# Heading\nContent"

            await service.ingest(
                file=upload,
                collection="meta-coll",
                metadata={"author": "alice", "project": "demo"},
            )

    docs = mock_vector_store.add_documents.call_args[0][1]
    assert docs[0].metadata["author"] == "alice"
    assert docs[0].metadata["project"] == "demo"
    assert docs[0].metadata["document_id"] == "meta-id"


# ---------------------------------------------------------------------------
# ingest() -- metadata defaults to empty dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_metadata_defaults_to_empty(service, mock_vector_store):
    """When metadata is None, an empty dict is used (no KeyError)."""
    upload = _make_upload_file("file.txt", b"text")

    with (
        patch("app.services.document_service.tempfile") as mock_tempfile,
        patch("app.services.document_service.os") as mock_os,
        patch("app.services.document_service.RecursiveCharacterTextSplitter") as MockSplitter,
        patch("app.services.document_service.uuid") as mock_uuid,
    ):
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/x.txt"
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp
        mock_os.path.splitext.return_value = ("file", ".txt")

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["chunk"]
        MockSplitter.return_value = mock_splitter_instance
        mock_uuid.uuid4.return_value = "noid"

        with patch.object(service, "_parse_file", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "text"

            await service.ingest(
                file=upload,
                collection="c",
                metadata=None,
            )

    docs = mock_vector_store.add_documents.call_args[0][1]
    # Only standard keys present, no extra metadata
    assert "author" not in docs[0].metadata
    assert docs[0].metadata["document_id"] == "noid"


# ---------------------------------------------------------------------------
# ingest() -- cleanup happens even on error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_error_still_cleans_up(service):
    """If an error occurs during parsing, the temp file is still removed."""
    upload = _make_upload_file("bad.pdf", b"corrupted")

    with (
        patch("app.services.document_service.tempfile") as mock_tempfile,
        patch("app.services.document_service.os") as mock_os,
        patch("app.services.document_service.uuid"),
    ):
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/bad.pdf"
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp
        mock_os.path.splitext.return_value = ("bad", ".pdf")

        with patch.object(
            service,
            "_parse_file",
            new_callable=AsyncMock,
            side_effect=RuntimeError("parse failed"),
        ):
            with pytest.raises(RuntimeError, match="parse failed"):
                await service.ingest(file=upload, collection="c")

    # Cleanup still happened
    mock_os.unlink.assert_called_once_with("/tmp/bad.pdf")


# ---------------------------------------------------------------------------
# _parse_file() -- routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_file_routes_pdf(service):
    """suffix='.pdf' routes to _parse_pdf."""
    with patch.object(service, "_parse_pdf", return_value="pdf text") as mock_pdf:
        result = await service._parse_file("/tmp/doc.pdf", ".pdf")

    mock_pdf.assert_called_once_with("/tmp/doc.pdf")
    assert result == "pdf text"


@pytest.mark.asyncio
async def test_parse_file_routes_txt(service):
    """suffix='.txt' reads the file directly."""
    m = mock_open(read_data="plain text content")
    with patch("builtins.open", m):
        result = await service._parse_file("/tmp/doc.txt", ".txt")

    m.assert_called_once_with("/tmp/doc.txt", encoding="utf-8")
    assert result == "plain text content"


@pytest.mark.asyncio
async def test_parse_file_routes_md(service):
    """suffix='.md' reads the file directly (same as .txt)."""
    m = mock_open(read_data="# Markdown")
    with patch("builtins.open", m):
        result = await service._parse_file("/tmp/readme.md", ".md")

    assert result == "# Markdown"


@pytest.mark.asyncio
async def test_parse_file_routes_docx(service):
    """suffix='.docx' routes to _parse_docx."""
    with patch.object(service, "_parse_docx", return_value="docx text") as mock_docx:
        result = await service._parse_file("/tmp/doc.docx", ".docx")

    mock_docx.assert_called_once_with("/tmp/doc.docx")
    assert result == "docx text"


@pytest.mark.asyncio
async def test_parse_file_uppercase_suffix(service):
    """Suffix is lowercased before routing."""
    with patch.object(service, "_parse_pdf", return_value="pdf") as mock_pdf:
        result = await service._parse_file("/tmp/doc.pdf", ".PDF")

    mock_pdf.assert_called_once_with("/tmp/doc.pdf")
    assert result == "pdf"


@pytest.mark.asyncio
async def test_parse_file_unknown_suffix_falls_back_to_text(service):
    """Unknown suffix falls back to reading as text."""
    m = mock_open(read_data="csv,data")
    with patch("builtins.open", m):
        result = await service._parse_file("/tmp/data.csv", ".csv")

    assert result == "csv,data"


# ---------------------------------------------------------------------------
# _parse_pdf() -- pdfplumber
# ---------------------------------------------------------------------------


def test_parse_pdf_with_pdfplumber(service):
    """_parse_pdf uses pdfplumber to extract text from pages."""
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page one text"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page two text"
    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = None  # empty page

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch.dict("sys.modules", {"pdfplumber": MagicMock()}):
        import sys

        mock_pdfplumber = sys.modules["pdfplumber"]
        mock_pdfplumber.open.return_value = mock_pdf

        result = service._parse_pdf("/tmp/test.pdf")

    assert "Page one text" in result
    assert "Page two text" in result
    # Page 3 text was None, so it should not appear
    assert result == "Page one text\n\nPage two text"


def test_parse_pdf_fallback_to_pypdf(service):
    """When pdfplumber is not available, falls back to PyPDFLoader."""
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Fallback page 1"
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Fallback page 2"

    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc1, mock_doc2]

    with (
        patch.dict("sys.modules", {"pdfplumber": None}),
        patch(
            "langchain_community.document_loaders.PyPDFLoader",
            return_value=mock_loader,
        ),
    ):
        # Force ImportError for pdfplumber
        with patch("builtins.__import__", side_effect=_import_blocker("pdfplumber")):
            result = service._parse_pdf("/tmp/test.pdf")

    assert "Fallback page 1" in result
    assert "Fallback page 2" in result


def _import_blocker(blocked_name):
    """Return a side_effect for __import__ that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"No module named '{blocked_name}'")
        return real_import(name, *args, **kwargs)

    return _blocker


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_calls_vector_store(service, mock_vector_store):
    """delete() gets collection and calls delete with document_id filter."""
    mock_collection = MagicMock()
    mock_vector_store.get_or_create_collection.return_value = mock_collection

    await service.delete("my-collection", "doc-123")

    mock_vector_store.get_or_create_collection.assert_called_once_with("my-collection")
    mock_collection.delete.assert_called_once_with(where={"document_id": "doc-123"})


@pytest.mark.asyncio
async def test_delete_different_collection(service, mock_vector_store):
    """delete() works with any collection name."""
    mock_collection = MagicMock()
    mock_vector_store.get_or_create_collection.return_value = mock_collection

    await service.delete("other-coll", "doc-456")

    mock_vector_store.get_or_create_collection.assert_called_once_with("other-coll")
    mock_collection.delete.assert_called_once_with(where={"document_id": "doc-456"})


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_uses_provided_vector_store():
    """When a VectorStoreManager is provided, it is used directly."""
    vs = MagicMock()

    from app.services.document_service import DocumentService

    svc = DocumentService(vector_store=vs)

    assert svc.vector_store is vs


def test_constructor_creates_default_vector_store():
    """When no VectorStoreManager is provided, a new one is created."""
    mock_vs = MagicMock()
    with patch(
        "app.services.document_service.VectorStoreManager",
        return_value=mock_vs,
    ):
        from app.services.document_service import DocumentService

        svc = DocumentService(vector_store=None)

    assert svc.vector_store is mock_vs


# ---------------------------------------------------------------------------
# ingest() -- file with no filename
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_file_with_no_filename(service, mock_vector_store):
    """UploadFile.filename is None -> suffix becomes empty string."""
    upload = MagicMock()
    upload.filename = None
    upload.read = AsyncMock(return_value=b"content")

    with (
        patch("app.services.document_service.tempfile") as mock_tempfile,
        patch("app.services.document_service.os") as mock_os,
        patch("app.services.document_service.RecursiveCharacterTextSplitter") as MockSplitter,
        patch("app.services.document_service.uuid") as mock_uuid,
    ):
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/noname"
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp
        mock_os.path.splitext.return_value = ("", "")

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["chunk"]
        MockSplitter.return_value = mock_splitter_instance
        mock_uuid.uuid4.return_value = "no-fn-id"

        with patch.object(service, "_parse_file", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "content"

            result = await service.ingest(file=upload, collection="c")

    # suffix was ""
    mock_parse.assert_awaited_once_with("/tmp/noname", "")
    assert result["filename"] is None

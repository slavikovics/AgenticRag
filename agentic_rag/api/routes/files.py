import asyncio
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agentic_rag.utils.file_parser import process_file_to_documents

from ..dependencies import get_qdrant_manager
from ..models import FilesUploadResponse, FileUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".pdf", ".csv", ".json"}


@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
):
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {ALLOWED_EXTENSIONS}",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            documents = process_file_to_documents(
                file_path=tmp_path,
                source_name=file.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            retriever = await get_qdrant_manager()
            count = await retriever.upsert_documents(documents)

            return FileUploadResponse(
                status="success",
                filename=file.filename,
                documents_indexed=count,
                chunks=len(documents),
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/upload-batch", response_model=FilesUploadResponse)
async def upload_files_batch(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
):
    try:
        details = []
        total_documents = 0
        files_processed = 0

        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()

            if file_ext not in ALLOWED_EXTENSIONS:
                details.append(
                    {
                        "filename": file.filename,
                        "status": "skipped",
                        "reason": f"Unsupported file type: {file_ext}",
                    }
                )
                continue

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_ext
                ) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                try:
                    documents = process_file_to_documents(
                        file_path=tmp_path,
                        source_name=file.filename,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    retriever = await get_qdrant_manager()
                    count = await retriever.upsert_documents(documents)

                    details.append(
                        {
                            "filename": file.filename,
                            "status": "success",
                            "documents_indexed": count,
                            "chunks": len(documents),
                        }
                    )

                    total_documents += count
                    files_processed += 1
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                details.append(
                    {
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return FilesUploadResponse(
            status="completed",
            files_processed=files_processed,
            total_documents_indexed=total_documents,
            details=details,
        )

    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        retriever = await get_qdrant_manager()
        count = await retriever.delete_by_source(filename)

        return {
            "status": "success",
            "filename": filename,
            "documents_deleted": count,
        }

    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/upload-from-directory")
async def upload_files_from_directory(
    directory_path: str = Form(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    recursive: bool = Form(default=False),
    min_chunks_count: int = Form(default=300),
    max_concurrent_uploads: int = Form(default=50),  # new param
):
    try:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise HTTPException(status_code=400, detail="Invalid directory path")

        # --- Collect files ---
        files_to_process = []
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                        files_to_process.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if (
                    os.path.isfile(file_path)
                    and os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS
                ):
                    files_to_process.append(file_path)

        if not files_to_process:
            return {
                "status": "completed",
                "files_processed": 0,
                "total_documents_indexed": 0,
                "details": [],
                "message": "No supported files found",
            }

        # --- Process each file into chunks (CPU-bound, keep sequential) ---
        all_chunks: list[tuple[str, list]] = []  # (filename, documents)
        details = []

        for file_path in files_to_process:
            try:
                documents = process_file_to_documents(
                    file_path=file_path,
                    source_name=os.path.basename(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                all_chunks.append((os.path.basename(file_path), documents))
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                details.append(
                    {
                        "filename": os.path.basename(file_path),
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # --- Build batches of min_chunks_count ---
        batches: list[list[tuple[str, list]]] = []
        current_batch: list[tuple[str, list]] = []
        current_batch_size = 0

        for filename, documents in all_chunks:
            current_batch.append((filename, documents))
            current_batch_size += len(documents)
            if current_batch_size >= min_chunks_count:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

        if current_batch:
            batches.append(current_batch)

        # --- Upload batches concurrently ---
        semaphore = asyncio.Semaphore(max_concurrent_uploads)
        total_documents_indexed = 0
        files_processed = 0

        async def upload_batch(batch: list[tuple[str, list]]) -> dict:
            async with semaphore:
                batch_docs = [doc for _, docs in batch for doc in docs]
                filenames = [fname for fname, _ in batch]
                try:
                    retriever = await get_qdrant_manager()
                    count = await retriever.upsert_documents(batch_docs)
                    logger.info(
                        f"Uploaded batch of {count} docs from {len(filenames)} files"
                    )
                    return {"filenames": filenames, "count": count, "error": None}
                except Exception as e:
                    logger.error(f"Batch upload failed for {filenames}: {e}")
                    return {"filenames": filenames, "count": 0, "error": str(e)}

        results = await asyncio.gather(*[upload_batch(b) for b in batches])

        # --- Collate results into details ---
        for result in results:
            files_processed += len(result["filenames"])
            total_documents_indexed += result["count"]
            for fname in result["filenames"]:
                details.append(
                    {
                        "filename": fname,
                        "status": "processed" if not result["error"] else "failed",
                        "documents_indexed": result["count"],
                        **({"error": result["error"]} if result["error"] else {}),
                    }
                )

        return {
            "status": "completed",
            "files_processed": files_processed,
            "total_documents_indexed": total_documents_indexed,
            "min_chunks_required": min_chunks_count,
            "batches_uploaded": len(batches),
            "details": details,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Directory upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

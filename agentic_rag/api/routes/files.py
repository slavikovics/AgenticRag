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
):
    try:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise HTTPException(status_code=400, detail="Invalid directory path")

        files_to_process = []

        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in ALLOWED_EXTENSIONS:
                        files_to_process.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in ALLOWED_EXTENSIONS:
                        files_to_process.append(file_path)

        details = []
        total_documents = 0
        files_processed = 0

        for file_path in files_to_process:
            try:
                documents = process_file_to_documents(
                    file_path=file_path,
                    source_name=os.path.basename(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                retriever = await get_qdrant_manager()
                count = await retriever.upsert_documents(documents)

                details.append(
                    {
                        "filename": os.path.basename(file_path),
                        "status": "success",
                        "documents_indexed": count,
                    }
                )

                logger.info(f"File {os.path.basename(file_path)} uploaded successfully")

                total_documents += count
                files_processed += 1
            except Exception as e:
                details.append(
                    {
                        "filename": os.path.basename(file_path),
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return {
            "status": "completed",
            "files_processed": files_processed,
            "total_documents_indexed": total_documents,
            "details": details,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Directory upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

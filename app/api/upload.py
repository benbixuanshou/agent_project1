from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
import aiofiles
import os

from app.config import settings
from app.models.schemas import ApiResponse, FileUploadResponse

router = APIRouter(tags=["upload"])

ALLOWED_EXTENSIONS = {ext.strip() for ext in settings.upload_allowed_extensions.split(",")}


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "Filename cannot be empty"})

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file format, only: {settings.upload_allowed_extensions}"},
        )

    upload_dir = os.path.normpath(settings.upload_path)
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    # Save file
    content = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Auto-index into Milvus
    indexing_error = None
    try:
        from app.rag.indexer import IndexingService
        indexer = IndexingService(
            vector_store=request.app.state.vector_store,
            embedder=request.app.state.embedder,
        )
        await indexer.index_file(file_path)
    except Exception as e:
        indexing_error = str(e)

    response = FileUploadResponse(
        file_name=file.filename,
        file_path=file_path,
        file_size=len(content),
    )
    if indexing_error:
        return JSONResponse(
            status_code=207,
            content=ApiResponse(
                code=207,
                message=f"File uploaded but indexing failed: {indexing_error}",
                data=response,
            ).model_dump(),
        )
    return ApiResponse(data=response)

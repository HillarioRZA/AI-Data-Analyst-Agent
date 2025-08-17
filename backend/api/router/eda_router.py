from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import io
from backend.services.eda import main as eda_main
from backend.services.agent import main as agent_main


router = APIRouter(
    prefix="/api/eda", 
    tags=["EDA"] 
)

@router.post("/describe")
async def describe_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV.")
    contents = await file.read()
    description = eda_main.get_csv_description(contents)
    if description is None:
        raise HTTPException(status_code=500, detail="Gagal memproses file CSV.")
    return {
        "filename": file.filename,
        "statistics": description
    }

@router.post("/correlation-heatmap", response_class=StreamingResponse)
async def get_correlation_heatmap(file: UploadFile = File(...)):
    """
    Endpoint untuk menghasilkan gambar heatmap korelasi dari file CSV.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV.")
    
    contents = await file.read()
    image_bytes = eda_main.generate_correlation_heatmap(contents)
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Gagal membuat heatmap.")

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@router.post("/histogram", response_class=StreamingResponse)
async def get_histogram(
    file: UploadFile = File(...),
    column_name: str = Form(...)
):

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")

    contents = await file.read()

    result = eda_main.generate_histogram(contents, column_name)

    if result == "column_not_found":
        raise HTTPException(status_code=404, detail=f"Kolom '{column_name}' tidak ditemukan di dalam file.")
    if result == "column_not_numeric":
        raise HTTPException(status_code=400, detail=f"Kolom '{column_name}' bukan numerik dan tidak bisa dibuatkan histogram.")
    if result is None:
        raise HTTPException(status_code=500, detail="Gagal membuat histogram.")

    return StreamingResponse(io.BytesIO(result), media_type="image/png")

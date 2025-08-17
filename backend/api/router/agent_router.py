import io
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from backend.services.eda import main as eda_main
from backend.services.agent import main as agent_main

router = APIRouter(
    prefix="/api/agent",
    tags=["Agent"]
)

@router.post("/decide")
def decide_action(prompt: str = Body(..., embed=True)):
    """
    Endpoint untuk menerima prompt pengguna dan membiarkan AI Agent
    memutuskan tindakan/tool yang akan digunakan.
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt tidak boleh kosong.")

    plan = agent_main.get_agent_plan(prompt)

    if "error" in plan:
        raise HTTPException(status_code=500, detail=plan["error"])

    return plan

@router.post("/execute")
async def execute_action(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Menerima file dan prompt, lalu secara otomatis menjalankan
    seluruh alur kerja ReAct: Reasoning -> Acting.
    """
    # 1. Pastikan file adalah CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")
        
    # 2. Baca konten file terlebih dahulu
    contents = await file.read()
    
    # 3. Reasoning: Dapatkan rencana dari AI Agent
    plan = agent_main.get_agent_plan(prompt)
    if "error" in plan:
        raise HTTPException(status_code=500, detail=f"Agent gagal membuat rencana: {plan.get('detail')}")

    tool_to_use = plan.get("tool_name")
    
    # 4. Acting: Jalankan tool yang dipilih berdasarkan rencana
    if tool_to_use == "describe":
        result = eda_main.get_csv_description(contents)
        if result is None:
            raise HTTPException(status_code=500, detail="Gagal menjalankan tool 'describe'.")
        return {"filename": file.filename, "analysis_result": result}
        
    elif tool_to_use == "correlation-heatmap":
        image_bytes = eda_main.generate_correlation_heatmap(contents)
        if image_bytes is None:
            raise HTTPException(status_code=500, detail="Gagal menjalankan tool 'correlation-heatmap'.")
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
        
    elif tool_to_use == "histogram":
        column_name = plan.get("column_name")
        if not column_name:
            raise HTTPException(status_code=400, detail="Agent tidak bisa menentukan nama kolom untuk histogram.")
            
        result = eda_main.generate_histogram(contents, column_name)
        if isinstance(result, str): # Penanganan error dari service
            if result == "column_not_found":
                raise HTTPException(status_code=404, detail=f"Kolom '{column_name}' tidak ditemukan.")
            if result == "column_not_numeric":
                raise HTTPException(status_code=400, detail=f"Kolom '{column_name}' bukan numerik.")
        if result is None:
             raise HTTPException(status_code=500, detail="Gagal menjalankan tool 'histogram'.")

        return StreamingResponse(io.BytesIO(result), media_type="image/png")
        
    else:
        raise HTTPException(status_code=400, detail=f"Tool '{tool_to_use}' tidak dikenali.")
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from app.cv_pipeline import process_video_dummy

app = FastAPI(title="Road Sign Recognition API")

# for creating directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.post("/detect/")
async def detect_signs(file: UploadFile = File(...)):
    # 1. Saving the uploaded video file to the server
    input_path = f"uploads/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_path = f"results/processed_{file.filename}"
    
    # 2. Calling the video processing function (placeholder without actual neural network)
    stats = process_video_dummy(input_path, output_path)
    
    # 3. Returning JSON with results
    return JSONResponse(content={
        "status": "success",
        "message": "Video processed successfully.",
        "output_file": output_path,
        "statistics": stats
    })
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import numpy as np
import os
app = FastAPI()
if not os.path.exists("outputs"):
    os.makedirs("outputs")
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html") as f:
        return f.read()
@app.post("/process-image")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_path = f"outputs/edited_{file.filename}"
    cv2.imwrite(output_path, processed_img)
    return FileResponse(output_path)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
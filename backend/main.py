from fastapi import FastAPI, UploadFile, File,  HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from docx import Document
import io


app = FastAPI()

#Cors Settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]

#Cors Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of allowed origins
    allow_credentials=True,      # Allow cookies to be included in requests
    allow_methods=["*"],         # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],         # Allow all headers
)



# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}

@app.get("/api/data")
async def data():
    return {"message": "Retrieved Successfully"}

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    extracted_text = ''
    file_content = await file.read()
    
    if file.content_type == "application/pdf":
        try:
            pdf_reader = PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + '\n'
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error Processing pdf:{e}")
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            # Use io.BytesIO to treat bytes content as a file-like object
            document = Document(io.BytesIO(file_content))
            for paragraph in document.paragraphs:
                extracted_text += paragraph.text + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing DOCX: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX.")


    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "extracted_text": extracted_text.strip() # Remove leading/trailing whitespace
    }


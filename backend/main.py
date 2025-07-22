from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
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




@app.get("/hello-world")
async def hello_world():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.get("/api/data")
async def data():
    return {"message": "Retrieved Successfully "}
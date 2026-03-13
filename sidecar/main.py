from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import torch
import shutil
import uuid
import gc
import re
import psutil
from dotenv import load_dotenv

# Try to find .env in parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

def log_memory(label):
    try:
        process = psutil.Process()
        mem_rss = process.memory_info().rss / (1024 * 1024)
        mem_vms = process.memory_info().vms / (1024 * 1024)
        print(f"[{label}] RSS: {mem_rss:.1f} MB | VMS: {mem_vms:.1f} MB")
    except:
        pass

# Simplified Regex to avoid Catastrophic Backtracking
PROPER_NOUN_REGEX = re.compile(r'\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)?\b') # Max 2 words, no nested loops
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
DATE_REGEX = re.compile(r'\b\d{4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b')

class DoclingProcessor:
    def __init__(self):
        self.converter = None

    def _ensure_converter(self):
        if self.converter is None:
            # Lazy import to save memory
            print("Importing Docling (Heavy)...")
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()

    def process(self, file_path: str) -> str:
        if file_path.lower().endswith((".md", ".txt")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        self._ensure_converter()
        try:
            result = self.converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            if file_path.lower().endswith((".md", ".txt")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            raise e

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        text = text[:8000] # Limit chunk size for NER
        entities = {
            "proper_nouns": list(set(PROPER_NOUN_REGEX.findall(text)))[:20],
            "dates": DATE_REGEX.findall(text)[:10],
            "emails": EMAIL_REGEX.findall(text)[:5]
        }
        return entities

app = FastAPI(title="1C1 ML Sidecar")
doc_processor = DoclingProcessor()

# Global cache for heavy models
MODELS = {}

def get_model(model_name: str, task: str):
    key = f"{task}_{model_name}"
    if key not in MODELS:
        device = os.getenv("DEVICE", "cpu")
        engine = os.getenv("INFERENCE_ENGINE", "torch")
        print(f"Loading {task} model: {model_name} on {device} (engine: {engine})...")
        
        if task == "embedding":
            from sentence_transformers import SentenceTransformer
            if engine == "onnx":
                # Only use ONNX if explicitly requested
                MODELS[key] = SentenceTransformer(model_name, device=device, backend="onnx")
            else:
                MODELS[key] = SentenceTransformer(model_name, device=device)
        else: # reranker
            from sentence_transformers import CrossEncoder
            MODELS[key] = CrossEncoder(model_name, device=device)
            
    return MODELS[key]

class EncodeRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class EntityBatchRequest(BaseModel):
    texts: List[str]

@app.post("/encode")
def encode(req: EncodeRequest):
    log_memory("Before Encode")
    model = get_model(os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base"), "embedding")
    
    # Prefix required for E5
    prefixed = [f"passage: {t}" for t in req.texts]
    
    # Internal batching to keep it stable
    embeddings = model.encode(prefixed, show_progress_bar=False, batch_size=4).tolist()
    
    log_memory("After Encode")
    return {"embeddings": embeddings}

@app.post("/rerank")
def rerank(req: RerankRequest):
    log_memory("Before Rerank")
    model = get_model(os.getenv("RERANKER_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"), "reranker")
    
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs, batch_size=4)
    
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    
    log_memory("After Rerank")
    return {"scores": scores}

@app.post("/extract_entities_batch")
def extract_entities_batch(req: EntityBatchRequest):
    results = []
    for t in req.texts:
        results.append(doc_processor.extract_entities(t))
    return {"results": results}

@app.post("/parse_pdf")
def parse_pdf(file: UploadFile = File(...)):
    filename = file.filename
    temp_filename = f"temp_{uuid.uuid4()}_{filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = doc_processor.process(temp_filename)
        return {"text": text}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Use single worker to minimize footprint
    print("Starting Sidecar on 127.0.0.1:50051...")
    uvicorn.run(app, host="127.0.0.1", port=50051, workers=1)

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

# Simplified Regex for baseline extraction
PROPER_NOUN_REGEX = re.compile(r'\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)?\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
DATE_REGEX = re.compile(r'\b\d{4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b')

class DoclingProcessor:
    def __init__(self):
        self.converter = None

    def _ensure_converter(self):
        if self.converter is None:
            # Lazy import because Docling + LayoutLM models are massive
            print("Importing Docling (Heavy Artillery)...")
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()

    def process_heavy(self, file_path: str) -> str:
        """The expensive OCR/Layout parsing route."""
        if file_path.lower().endswith((".md", ".txt", ".html")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        self._ensure_converter()
        try:
            result = self.converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Docling failed, attempting raw read: {e}")
            if file_path.lower().endswith((".md", ".txt")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            raise e

    def process_smart(self, file_path: str) -> str:
        """The cheap, fast extraction route."""
        if file_path.lower().endswith((".md", ".txt", ".html")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif file_path.lower().endswith(".pdf"):
            try:
                import pypdf
                text = ""
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                print("Warning: pypdf not found. Falling back to raw read.")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        return ""

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        # Limit context window for regex extraction speed
        text = text[:8000]
        entities = {
            "proper_nouns": list(set(PROPER_NOUN_REGEX.findall(text)))[:20],
            "dates": DATE_REGEX.findall(text)[:10],
            "emails": EMAIL_REGEX.findall(text)[:5]
        }
        return entities

app = FastAPI(title="1C1 Headless Sidecar (ML)")
doc_processor = DoclingProcessor()

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
                MODELS[key] = SentenceTransformer(model_name, device=device, backend="onnx")
            else:
                MODELS[key] = SentenceTransformer(model_name, device=device)
        else: # reranker
            from sentence_transformers import CrossEncoder
            MODELS[key] = CrossEncoder(model_name, device=device)
            
    return MODELS[key]

# API Schemas
class EncodeRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class EntityBatchRequest(BaseModel):
    texts: List[str]

class SemanticChunkRequest(BaseModel):
    text: str
    threshold: float = 0.6  # Cosine similarity drop threshold
    max_words: int = 400    # Hard limit on chunk size

@app.post("/encode")
def encode(req: EncodeRequest):
    model = get_model(os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base"), "embedding")
    prefixed = [f"passage: {t}" for t in req.texts]
    # Small internal batch to keep RAM stable
    embeddings = model.encode(prefixed, show_progress_bar=False, batch_size=4).tolist()
    return {"embeddings": embeddings}

@app.post("/rerank")
def rerank(req: RerankRequest):
    model = get_model(os.getenv("RERANKER_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"), "reranker")
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs, batch_size=4)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return {"scores": scores}

@app.post("/extract_entities_batch")
def extract_entities_batch(req: EntityBatchRequest):
    results = [doc_processor.extract_entities(t) for t in req.texts]
    return {"results": results}

@app.post("/parse/heavy")
def parse_heavy(file: UploadFile = File(...)):
    """Endpoint 1: Swiss Watch Docling Parser (Expensive)"""
    filename = os.path.basename(file.filename)
    temp_filename = f"temp_heavy_{uuid.uuid4()}_{filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = doc_processor.process_heavy(temp_filename)
        return {"text": text, "method": "docling_heavy"}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        gc.collect()

@app.post("/parse/smart")
def parse_smart(file: UploadFile = File(...)):
    """Endpoint 2: Fast pypdf Extraction (Cheap)"""
    filename = os.path.basename(file.filename)
    temp_filename = f"temp_smart_{uuid.uuid4()}_{filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = doc_processor.process_smart(temp_filename)
        return {"text": text, "method": "pypdf_smart"}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        gc.collect()

@app.post("/chunk/semantic")
def semantic_chunk(req: SemanticChunkRequest):
    """
    Intelligent semantic chunker using sentence-transformers.
    Groups sentences together as long as their embeddings remain similar.
    """
    # 1. Very fast heuristic sentence split (covers 95% of cases)
    # Splits on punctuation followed by space, ensuring no empty strings.
    sentences_raw = re.split(r'(?<=[.!?]) +', req.text)
    sentences = [s.strip() for s in sentences_raw if len(s.strip()) > 15]
    
    if not sentences:
        return {"chunks": [req.text]}
    if len(sentences) == 1:
        return {"chunks": sentences}
        
    model = get_model(os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base"), "embedding")
    
    # Embed sentences individually (this is extremely fast on small strings)
    prefixed = [f"passage: {s}" for s in sentences]
    embeddings = model.encode(prefixed, show_progress_bar=False)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Compute cosine similarity between the previous sentence and the current one
        # This tells us if the topic/semantic context has suddenly shifted.
        sim = torch.nn.functional.cosine_similarity(embeddings[i-1:i], embeddings[i:i+1]).item()
        
        current_len = sum(len(s.split()) for s in current_chunk)
        next_len = len(sentences[i].split())
        
        # Merge if similarity is high and word limit isn't exceeded
        if sim >= req.threshold and (current_len + next_len) < req.max_words:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return {"chunks": chunks}

@app.get("/health")
def health():
    return {"status": "headless_online"}

if __name__ == "__main__":
    import uvicorn
    # Single worker to respect local hardware limits
    print("Starting Headless ML Sidecar on 127.0.0.1:50051...")
    uvicorn.run(app, host="127.0.0.1", port=50051, workers=1)
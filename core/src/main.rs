use axum::{
    routing::{get, post},
    Router, Json, extract::State, extract::Query,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::process::Command;
use std::collections::HashMap;
use tokio::sync::mpsc;

mod db;
use db::{VectorDB, Document, DocumentMeta};

struct IngestTask {
    docs: Vec<Document>,
    vectors: Vec<Vec<f32>>,
}

#[derive(Clone)]
struct AppState {
    db: Arc<VectorDB>,
    sidecar_url: String,
    http_client: reqwest::Client,
    ingest_tx: mpsc::Sender<IngestTask>,
}

#[derive(Serialize, Deserialize)]
struct SearchQuery {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default = "default_rerank")]
    rerank: bool,
}

fn default_limit() -> usize { 10 }
fn default_rerank() -> bool { true }

#[derive(Serialize, Deserialize)]
struct SearchResult {
    text: String,
    score: f32,
    metadata: DocumentMeta,
}

#[derive(Serialize, Deserialize)]
struct IngestItem {
    filename: String,
    text: String,
    title: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct EncodeRequest {
    texts: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct EncodeResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct RerankResponse {
    scores: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct EntityResponse {
    proper_nouns: Vec<String>,
    dates: Vec<String>,
    emails: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct EntityBatchRequest {
    texts: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct EntityBatchResponse {
    results: Vec<EntityResponse>,
}

async fn get_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "online",
        "mode": "headless",
        "sidecar": "running"
    }))
}

async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> Json<serde_json::Value> {
    // 1. Encode query via sidecar
    let enc_req = EncodeRequest { texts: vec![params.query.clone()] };
    let enc_res: EncodeResponse = state.http_client.post(format!("{}/encode", state.sidecar_url))
        .json(&enc_req)
        .send().await.unwrap().json().await.unwrap();
    let q_vec = &enc_res.embeddings[0];

    // 2. Concurrent Vector and FTS search
    let (vec_res, fts_res) = tokio::join!(
        state.db.search_vector(q_vec, params.limit * 5),
        state.db.search_fts(&params.query, params.limit * 5)
    );

    let vec_docs = vec_res.unwrap_or_default();
    let fts_docs = fts_res.unwrap_or_default();

    // 3. Reciprocal Rank Fusion
    let k = 60.0;
    let mut combined_scores: HashMap<u64, (f32, Document)> = HashMap::new();

    let mut process_hits = |hits: Vec<Document>| {
        for (rank, hit) in hits.into_iter().enumerate() {
            use std::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            hit.text.hash(&mut hasher);
            let key = hasher.finish();
            
            let entry = combined_scores.entry(key).or_insert((0.0, hit));
            entry.0 += 1.0 / (k + (rank as f32) + 1.0);
        }
    };
    process_hits(vec_docs);
    process_hits(fts_docs);

    let mut sorted_candidates: Vec<_> = combined_scores.into_iter().map(|(_, v)| v).collect();
    sorted_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    sorted_candidates.truncate(params.limit * 3);

    // 4. Rerank Phase
    if params.rerank && !sorted_candidates.is_empty() {
        let docs: Vec<String> = sorted_candidates.iter().map(|c| c.1.text.clone()).collect();
        let rr_req = RerankRequest { query: params.query.clone(), documents: docs };
        let rr_res_raw = state.http_client.post(format!("{}/rerank", state.sidecar_url))
            .json(&rr_req)
            .send().await;

        if let Ok(res) = rr_res_raw {
            if let Ok(rr_res) = res.json::<RerankResponse>().await {
                for (i, score) in rr_res.scores.iter().enumerate() {
                    if i < sorted_candidates.len() {
                        sorted_candidates[i].0 = *score;
                    }
                }
                sorted_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
    }

    sorted_candidates.truncate(params.limit);

    let results: Vec<SearchResult> = sorted_candidates.into_iter().map(|(score, doc)| {
        SearchResult {
            text: doc.text,
            score,
            metadata: doc.metadata,
        }
    }).collect();

    Json(serde_json::json!({
        "status": "success",
        "results": results
    }))
}

async fn ingest_batch(
    State(state): State<AppState>,
    Json(items): Json<Vec<IngestItem>>,
) -> Json<serde_json::Value> {
    let mut docs = Vec::new();
    let clean_texts: Vec<String> = items.iter().map(|item| {
        item.text.split_whitespace().collect::<Vec<&str>>().join(" ")
    }).collect();

    if clean_texts.is_empty() {
        return Json(serde_json::json!({"status": "success", "count": 0}));
    }

    // 1. Batch Entity Extraction
    let ent_req = EntityBatchRequest { texts: clean_texts.clone() };
    let ent_res_raw = state.http_client.post(format!("{}/extract_entities_batch", state.sidecar_url))
        .json(&ent_req)
        .send().await;
    
    let ent_res: EntityBatchResponse = match ent_res_raw {
        Ok(res) => res.json().await.unwrap_or(EntityBatchResponse { results: vec![] }),
        Err(_) => EntityBatchResponse { results: vec![] },
    };

    // 2. Batch Embedding
    let enc_req = EncodeRequest { texts: clean_texts.clone() };
    let enc_res_raw = state.http_client.post(format!("{}/encode", state.sidecar_url))
        .json(&enc_req)
        .send().await;

    let enc_res: EncodeResponse = match enc_res_raw {
        Ok(res) => res.json().await.unwrap_or(EncodeResponse { embeddings: vec![] }),
        Err(_) => EncodeResponse { embeddings: vec![] },
    };

    if enc_res.embeddings.is_empty() {
        return Json(serde_json::json!({"status": "error", "message": "ML Sidecar failed to encode batch"}));
    }

    // 3. Construct Documents
    for (i, item) in items.into_iter().enumerate() {
        let clean_text = &clean_texts[i];
        let entities_str = if i < ent_res.results.len() {
            let r = &ent_res.results[i];
            let mut all = r.proper_nouns.clone();
            all.extend(r.dates.clone());
            all.extend(r.emails.clone());
            all.join(" ")
        } else {
            "".to_string()
        };

        let doc = Document {
            text: clean_text.clone(),
            entities: entities_str,
            metadata: DocumentMeta {
                filename: item.filename.clone(),
                title: item.title.unwrap_or(item.filename),
                source: "batch_ingest".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            }
        };
        docs.push(doc);
    }

    let count = docs.len();
    if let Err(e) = state.ingest_tx.send(IngestTask { docs, vectors: enc_res.embeddings }).await {
        eprintln!("Failed to queue ingestion task: {}", e);
        return Json(serde_json::json!({"status": "error", "message": "Failed to queue for ingestion"}));
    }

    Json(serde_json::json!({
        "status": "success",
        "count": count
    }))
}

#[tokio::main]
async fn main() {
    println!("Starting OS Level processes...");

    // Spawn Sidecar.
    let python_path = if std::path::Path::new("../.venv/bin/python").exists() {
        "../.venv/bin/python"
    } else {
        "python3"
    };

    println!("Starting ML sidecar using {}...", python_path);
    let mut sidecar_process = Command::new(python_path)
        .arg("../sidecar/main.py")
        .spawn()
        .expect("Failed to start ML sidecar process");

    // Give sidecar a bit of time to load the massive ONNX models
    println!("Waiting 5s for Sidecar ONNX cold-start...");
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    // Load Database
    let db = Arc::new(VectorDB::new("../local_storage/lancedb_rust"));

    let (ingest_tx, mut ingest_rx) = mpsc::channel::<IngestTask>(100);
    
    let db_writer = Arc::clone(&db);
    tokio::spawn(async move {
        while let Some(task) = ingest_rx.recv().await {
            if let Err(e) = db_writer.add_documents(&task.docs, &task.vectors).await {
                eprintln!("Error writing to database: {}", e);
            } else {
                println!("Successfully ingested batch of {} documents.", task.docs.len());
            }
        }
    });

    let state = AppState {
        db,
        sidecar_url: "http://127.0.0.1:50051".to_string(),
        http_client: reqwest::Client::new(),
        ingest_tx,
    };

    let app = Router::new()
        .route("/status", get(get_status))
        .route("/search", get(search))
        .route("/ingest_batch", post(ingest_batch))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("Rust Core running on http://127.0.0.1:8000");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.unwrap();
            println!("Shutting down... killing sidecar");
            let _ = sidecar_process.kill().await;
        })
        .await
        .unwrap();
}

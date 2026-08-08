#[path = "../db.rs"]
mod db;

use db::{VectorDB, Document, DocumentMeta};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct ExportItem {
    id: String,
    text: String,
    vector: Vec<f32>,
    #[allow(dead_code)]
    chunk_index: usize,
    #[allow(dead_code)]
    metadata: Option<serde_json::Value>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let clean = args.iter().any(|arg| arg == "--clean");
    
    // Find jsonl file argument (skip binary path and --clean flag)
    let jsonl_path_str = args.iter()
        .skip(1)
        .find(|arg| !arg.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "../sovereign-engine/export.jsonl".to_string());
    
    let jsonl_path = PathBuf::from(jsonl_path_str);

    println!("==================================================");
    println!("     ONECERO.ONE: VECTOR & PAYLOAD IMPORT");
    println!("==================================================");
    println!("Input JSONL: {:?}", jsonl_path);
    println!("Clean DB:    {}", clean);

    let storage_uri = std::env::var("STORAGE_URI").unwrap_or_else(|_| "../local_storage/lancedb_rust".to_string());
    println!("Storage URI: {}", storage_uri);

    if clean {
        println!("* Cleaning existing database files at {}...", storage_uri);
        if std::path::Path::new(&storage_uri).exists() {
            std::fs::remove_dir_all(&storage_uri)?;
            println!("✔ Cleaned old database files.");
        }
    }

    if !jsonl_path.exists() {
        return Err(format!("Input JSONL file {:?} does not exist", jsonl_path).into());
    }

    println!("* Initializing VectorDB...");
    let db = VectorDB::new(&storage_uri);
    println!("✔ VectorDB ready.");

    let file = File::open(&jsonl_path)?;
    let reader = BufReader::new(file);

    let batch_size = 10000;
    let mut docs = Vec::with_capacity(batch_size);
    let mut vectors = Vec::with_capacity(batch_size);
    let mut total_imported = 0;
    
    let start_time = Instant::now();
    let mut batch_start_time = Instant::now();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }

        let item: ExportItem = serde_json::from_str(&line)?;
        
        let doc = Document {
            text: item.text,
            entities: "".to_string(),
            metadata: DocumentMeta {
                filename: item.id.clone(),
                title: item.id,
                source: "msmarco".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
        };

        docs.push(doc);
        vectors.push(item.vector);

        if docs.len() == batch_size {
            let count = docs.len();
            db.add_documents(&docs, &vectors).await?;
            total_imported += count;
            
            println!(
                "✔ Imported batch of {} docs in {:.2?} (Total: {}, Elapsed: {:.2?})",
                count,
                batch_start_time.elapsed(),
                total_imported,
                start_time.elapsed()
            );
            
            docs.clear();
            vectors.clear();
            batch_start_time = Instant::now();
        }
    }

    // Import remaining documents
    if !docs.is_empty() {
        let count = docs.len();
        db.add_documents(&docs, &vectors).await?;
        total_imported += count;
        println!(
            "✔ Imported final batch of {} docs in {:.2?} (Total: {})",
            count,
            batch_start_time.elapsed(),
            total_imported
        );
    }

    println!("==================================================");
    println!("✔ Import completed successfully!");
    println!("✔ Total documents imported: {}", total_imported);
    println!("✔ Total time: {:.2?}", start_time.elapsed());
    println!("==================================================");

    Ok(())
}

use arrow_array::{
    builder::{Float32Builder, FixedSizeListBuilder, StringBuilder, StructBuilder},
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::dataset::{Dataset, WriteParams, WriteMode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::Path;
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, TantivyDocument};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentMeta {
    pub filename: String,
    pub title: String,
    pub source: String,
    pub timestamp: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Document {
    pub text: String,
    pub entities: String,
    pub metadata: DocumentMeta,
}

pub struct VectorDB {
    uri: String,
    table_name: String,
    tantivy_index: Index,
    // Store fields for easy access during query and indexing
    fts_text: tantivy::schema::Field,
    fts_entities: tantivy::schema::Field,
    fts_filename: tantivy::schema::Field,
}

impl VectorDB {
    pub fn new(uri: &str) -> Self {
        // Initialize Tantivy Schema
        let mut schema_builder = Schema::builder();
        let fts_text = schema_builder.add_text_field("text", TEXT | STORED);
        let fts_entities = schema_builder.add_text_field("entities", TEXT | STORED);
        let fts_filename = schema_builder.add_text_field("filename", STRING | STORED);
        let tantivy_schema = schema_builder.build();

        // Create or open Tantivy index
        let tantivy_path = format!("{}/tantivy", uri);
        std::fs::create_dir_all(&tantivy_path).unwrap();
        let tantivy_dir = tantivy::directory::MmapDirectory::open(&tantivy_path).unwrap();
        
        let tantivy_index = Index::open_or_create(tantivy_dir, tantivy_schema).unwrap();

        Self {
            uri: uri.to_string(),
            table_name: "documents".to_string(),
            tantivy_index,
            fts_text,
            fts_entities,
            fts_filename,
        }
    }

    pub async fn add_documents(&self, docs: &[Document], vectors: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if docs.is_empty() {
            return Ok(());
        }
        
        if docs.len() != vectors.len() {
            return Err("Documents and vectors length mismatch".into());
        }
        
        for vec in vectors {
            if vec.len() != 768 {
                return Err("Vector dimension must be exactly 768".into());
            }
        }

        // --- 1. TANTIVY FTS INGESTION ---
        let mut index_writer: IndexWriter = self.tantivy_index.writer(50_000_000)?; // 50MB heap
        for doc in docs {
            let mut tantivy_doc = TantivyDocument::default();
            tantivy_doc.add_text(self.fts_text, &doc.text);
            tantivy_doc.add_text(self.fts_entities, &doc.entities);
            tantivy_doc.add_text(self.fts_filename, &doc.metadata.filename);
            index_writer.add_document(tantivy_doc)?;
        }
        index_writer.commit()?;

        // --- 2. LANCEDB VECTOR INGESTION (ARROW) ---
        // Define Arrow Schema
        let metadata_fields = vec![
            Field::new("filename", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("source", DataType::Utf8, false),
            Field::new("timestamp", DataType::Utf8, false),
        ];

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 768), false),
            Field::new("text", DataType::Utf8, false),
            Field::new("entities", DataType::Utf8, false),
            Field::new("metadata", DataType::Struct(metadata_fields.clone().into()), false),
            Field::new("timestamp", DataType::Utf8, false),
        ]));

        // Build columns
        let float_builder = Float32Builder::with_capacity(docs.len() * 768);
        let mut vector_builder = FixedSizeListBuilder::new(float_builder, 768);
        let mut text_builder = StringBuilder::new();
        let mut entities_builder = StringBuilder::new();
        
        let mut filename_builder = StringBuilder::new();
        let mut title_builder = StringBuilder::new();
        let mut source_builder = StringBuilder::new();
        let mut mt_timestamp_builder = StringBuilder::new();
        
        let mut root_timestamp_builder = StringBuilder::new();

        for (i, doc) in docs.iter().enumerate() {
            // Append Vector
            let vec_values = vector_builder.values();
            for val in &vectors[i] {
                vec_values.append_value(*val);
            }
            vector_builder.append(true);

            // Append Standard Texts
            text_builder.append_value(&doc.text);
            entities_builder.append_value(&doc.entities);
            root_timestamp_builder.append_value(&doc.metadata.timestamp);

            // Append Metadata Struct Fields
            filename_builder.append_value(&doc.metadata.filename);
            title_builder.append_value(&doc.metadata.title);
            source_builder.append_value(&doc.metadata.source);
            mt_timestamp_builder.append_value(&doc.metadata.timestamp);
        }

        let mut metadata_builder = StructBuilder::new(
            metadata_fields.clone(),
            vec![
                Box::new(filename_builder),
                Box::new(title_builder),
                Box::new(source_builder),
                Box::new(mt_timestamp_builder),
            ],
        );

        for _ in 0..docs.len() {
            metadata_builder.append(true);
        }

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(vector_builder.finish()),
                Arc::new(text_builder.finish()),
                Arc::new(entities_builder.finish()),
                Arc::new(metadata_builder.finish()),
                Arc::new(root_timestamp_builder.finish()),
            ],
        )?;

        // Write to LanceDB
        let dataset_path = format!("{}/{}", self.uri, self.table_name);
        let batches = vec![batch];
        
        use arrow_array::RecordBatchIterator;
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        
        if Path::new(&dataset_path).exists() {
            let mut params = WriteParams::default();
            params.mode = WriteMode::Append;
            Dataset::write(reader, &dataset_path, Some(params)).await?;
        } else {
            Dataset::write(reader, &dataset_path, None).await?;
        }

        // Commit Tantivy FTS only after LanceDB succeeds to maintain atomicity
        index_writer.commit()?;

        Ok(())
    }

    pub async fn search_vector(&self, vector: &[f32], limit: usize) -> Result<Vec<Document>, Box<dyn std::error::Error + Send + Sync>> {
        let dataset_path = format!("{}/{}", self.uri, self.table_name);
        if !Path::new(&dataset_path).exists() {
            return Ok(vec![]);
        }

        let dataset = Dataset::open(&dataset_path).await?;
        
        use arrow_array::Float32Array;
        let query_array = Float32Array::from(vector.to_vec());

        let mut scanner = dataset.scan();
        scanner.nearest("vector", &query_array, limit)?;
        let batches = scanner.try_into_stream().await?;
        
        use futures::stream::StreamExt;
        let mut results = Vec::new();
        
        let mut stream = batches;
        while let Some(batch_res) = stream.next().await {
            let batch: RecordBatch = batch_res?;
            
            let texts = batch.column_by_name("text").ok_or("Missing 'text' column")?.as_any().downcast_ref::<arrow_array::StringArray>().ok_or("Invalid 'text' column type")?;
            let entities = batch.column_by_name("entities").ok_or("Missing 'entities' column")?.as_any().downcast_ref::<arrow_array::StringArray>().ok_or("Invalid 'entities' column type")?;
            
            let meta_struct = batch.column_by_name("metadata").ok_or("Missing 'metadata' column")?.as_any().downcast_ref::<arrow_array::StructArray>().ok_or("Invalid 'metadata' column type")?;
            let fn_array = meta_struct.column_by_name("filename").ok_or("Missing 'filename' column")?.as_any().downcast_ref::<arrow_array::StringArray>().ok_or("Invalid 'filename' column type")?;
            let t_array = meta_struct.column_by_name("title").ok_or("Missing 'title' column")?.as_any().downcast_ref::<arrow_array::StringArray>().ok_or("Invalid 'title' column type")?;
            let s_array = meta_struct.column_by_name("source").ok_or("Missing 'source' column")?.as_any().downcast_ref::<arrow_array::StringArray>().ok_or("Invalid 'source' column type")?;
            let ts_array = meta_struct.column_by_name("timestamp").ok_or("Missing 'timestamp' column")?.as_any().downcast_ref::<arrow_array::StringArray>().ok_or("Invalid 'timestamp' column type")?;

            for i in 0..batch.num_rows() {
                results.push(Document {
                    text: texts.value(i).to_string(),
                    entities: entities.value(i).to_string(),
                    metadata: DocumentMeta {
                        filename: fn_array.value(i).to_string(),
                        title: t_array.value(i).to_string(),
                        source: s_array.value(i).to_string(),
                        timestamp: ts_array.value(i).to_string(),
                    }
                });
            }
        }

        Ok(results)
    }

    pub async fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<Document>, Box<dyn std::error::Error + Send + Sync>> {
        let reader = self.tantivy_index.reader()?;
        let searcher = reader.searcher();
        
        use tantivy::query::QueryParser;
        let query_parser = QueryParser::for_index(&self.tantivy_index, vec![self.fts_text, self.fts_entities]);
        let parsed_query = match query_parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => return Ok(vec![]),
        };

        use tantivy::collector::TopDocs;
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (_score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            
            let text = retrieved_doc.get_first(self.fts_text).and_then(|v: &tantivy::schema::OwnedValue| v.as_str()).unwrap_or("").to_string();
            let entities = retrieved_doc.get_first(self.fts_entities).and_then(|v: &tantivy::schema::OwnedValue| v.as_str()).unwrap_or("").to_string();
            let filename = retrieved_doc.get_first(self.fts_filename).and_then(|v: &tantivy::schema::OwnedValue| v.as_str()).unwrap_or("").to_string();

            // We mock the rest of the metadata for FTS hits since Tantivy is only used for RRF merging by text
            results.push(Document {
                text,
                entities,
                metadata: DocumentMeta {
                    filename,
                    title: "".to_string(),
                    source: "fts".to_string(),
                    timestamp: "".to_string(),
                }
            });
        }

        Ok(results)
    }
}

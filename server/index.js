import express from 'express';
import cors from 'cors';
import {
  loadModel,
  GTE_LARGE_FP16,
  embed,
} from "@tetherto/qvac-sdk";
import sqlite3InitModule from "@sqliteai/sqlite-wasm";
import seedData1000plus from "../dataset/seed-data-1000-plus.json" with { type: "json" };
import fs from "fs";

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Global variables for models and RAG
let embeddingModelId;
let db;

// Initialize embedding model
const initEmbeddingModel = async () => {
  console.log('📦 Loading embedding model...');
  embeddingModelId = await loadModel(GTE_LARGE_FP16, {
    modelType: "embeddings",
    onProgress: (progress) => {
      console.log(`Loading model... ${Math.round(progress.percentage * 100)}%`);
    },
  });
  console.log('✅ Embedding model loaded successfully');
};

// Initialize vector database
const initVectorDb = async () => {
  console.log('🗄️  Initializing SQLite vector database...');
  const sqlite3 = await sqlite3InitModule();
  db = new sqlite3.oo1.DB(":memory:", "c");

  // Create table for documents with vector storage
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      prompt TEXT NOT NULL,
      expected_output TEXT NOT NULL,
      embedding BLOB NOT NULL
    )
  `);

  console.log("📚 Embedding documents...");
  let count = 0;
  const newSamples = []
  for (const sample of seedData1000plus) {
    const embedding = await embed(embeddingModelId, sample.prompt);
    db.exec({
      sql: "INSERT INTO documents VALUES (?, ?, ?, vector_as_f32(?))",
      bind: [
        sample.id,
        sample.prompt,
        JSON.stringify(sample.expected_output),
        JSON.stringify(embedding),
      ],
    });
    count++;
    newSamples.push({ ...sample, embedding });
  }
  fs.writeFileSync("samplesWithEmbeddings.json", JSON.stringify(newSamples, null, 2));

  // Initialize and optimize vector index
  db.exec(
    `SELECT vector_init('documents', 'embedding', 'type=FLOAT32,dimension=1024')`
  );

  // Quantize vectors
  db.exec(`SELECT vector_quantize('documents', 'embedding')`);

  // Preload quantized vectors in memory for optimal performance
  db.exec(`SELECT vector_quantize_preload('documents', 'embedding')`);
  
  console.log(`✅ ${count} documents embedded and indexed successfully`);
};

// Perform vector search
const performVectorSearch = async (query) => {
  const queryEmbedding = await embed(embeddingModelId, query);
  
  const results = [];
  
  db.exec({
    sql: `
      SELECT d.id, d.prompt, d.expected_output, v.distance 
      FROM documents d
      JOIN vector_quantize_scan('documents', 'embedding', vector_as_f32(?), 3) v
      ON d.rowid = v.rowid
    `,
    bind: [JSON.stringify(queryEmbedding)],
    rowMode: "object",
    callback: (row) => {
      row.expected_output = JSON.parse(row.expected_output);
      results.push(row);
    },
  });
  
  return results;
};

// Initialize everything on server startup
const initialize = async () => {
  try {
    await initEmbeddingModel();
    await initVectorDb();
    console.log('🚀 RAG system initialized successfully');
  } catch (error) {
    console.error('❌ Initialization error:', error);
    process.exit(1);
  }
};

// Routes
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    message: 'Server is running'
  });
});

app.post('/query', async (req, res) => {
  try {
    const { query } = req.body;

    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({ 
        error: 'Query parameter is required and must be a non-empty string' 
      });
    }

    console.log(`🔎 Searching for: "${query}"`);
    const results = await performVectorSearch(query);

    res.json({
      query,
      results,
      count: results.length
    });
  } catch (error) {
    console.error('❌ Query error:', error);
    res.status(500).json({ 
      error: 'An error occurred while processing your query',
      details: error.message 
    });
  }
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    error: 'Endpoint not found',
    availableEndpoints: [
      { method: 'GET', path: '/health', description: 'Check server health' },
      { method: 'POST', path: '/query', description: 'Query documents', body: { query: 'string' } }
    ]
  });
});

// Start server
const startServer = async () => {
  console.log('🔄 Initializing RAG system...');
  await initialize();
  
  app.listen(PORT, () => {
    console.log(`\n🌟 Server is running on http://localhost:${PORT}`);
    console.log(`📍 POST to http://localhost:${PORT}/query with body: { "query": "your search query" }`);
  });
};

startServer().catch(console.error);


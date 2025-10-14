import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import sqlite3InitModule from "@sqliteai/sqlite-wasm";
import seedData1000plus from "./meal-datasets/seed-data-1000-plus.json" with { type: "json" };
import mealDatasetOriginal from "./meal-datasets/meal-dataset-original.json" with { type: "json" };
import { calculatePayloadMetrics, extractJSON, writeResultIncrementally } from "../utils.js";

const responseSchema = z.object({
  payload: z
    .object({
      description: z.string(),
      calories: z.number(),
      carbsGrams: z.number(),
      proteinGram: z.number(),
      fatGram: z.number(),
      glycemicIndex: z.number(),
    })
    .optional(),
  error: z.string().optional(),
});

function mealPrompt(schema, top3) {
  // Build RAG examples section
  let ragExamples = "";
  if (top3 && top3.length > 0) {
    ragExamples = "\n--- SIMILAR EXAMPLES (Reference only) ---\n";
    top3.forEach((example, idx) => {
      ragExamples += `\nExample ${idx + 1} (distance: ${example.distance?.toFixed(1) || 'N/A'}):\nInput: "${example.prompt}"\nOutput: ${JSON.stringify(example.expected_output)}\n`;
    });
    ragExamples += "\n";
  }

  return `/no_think
Task: Parse meal logging queries into structured JSON format.

⚠️ CRITICAL RULE - READ FIRST:
If the query contains ANY identifiable food name (pasta, burger, nuts, eggs, apple, steak, chicken, etc.), 
you MUST return PAYLOAD with estimates. Do NOT be influenced by Similar Examples that show errors 
unless they have the EXACT or SIMILAR food items.

Examples:
- "Ate pasta" has identifiable food "pasta" → MUST return PAYLOAD
- "Had a burger" has identifiable food "burger" → MUST return PAYLOAD  
- "Ate some food" has NO identifiable food → return ERROR
- "Restaurant meal" has NO identifiable food → return ERROR

Output Schema:
${JSON.stringify(z.toJSONSchema(schema), null, 2)}

--- BASELINE EXAMPLES ---

Example A - Specific meal with details:
Input: "Lunch was tuna sandwich on white bread with lettuce and mayo"
Output: {"payload":{"description":"Tuna sandwich on white bread with lettuce and mayo","calories":380,"carbsGrams":32,"proteinGram":22,"fatGram":18,"glycemicIndex":58}}

Example B - Simple snack with identifiable foods:
Input: "String cheese and apple slices"
Output: {"payload":{"description":"String cheese and apple slices","calories":175,"carbsGrams":20,"proteinGram":7,"fatGram":7,"glycemicIndex":30}}

Example C - Single identifiable food (estimate portions):
Input: "Ate pasta"
Output: {"payload":{"description":"Pasta (1 cup cooked, estimated)","calories":220,"carbsGrams":43,"proteinGram":8,"fatGram":1,"glycemicIndex":60}}

Example D - Single word food:
Input: "Ate eggs"
Output: {"payload":{"description":"Eggs (2 large, estimated)","calories":140,"carbsGrams":1,"proteinGram":12,"fatGram":10,"glycemicIndex":0}}

Example E - Beverage with quantity:
Input: "Coconut water (500ml)"
Output: {"payload":{"description":"Coconut water (500ml)","calories":90,"carbsGrams":22,"proteinGram":2,"fatGram":0,"glycemicIndex":35}}

Example F - Generic branded item:
Input: "Ate a protein bar"
Output: {"payload":{"description":"Protein bar (standard, estimated)","calories":200,"carbsGrams":24,"proteinGram":10,"fatGram":7,"glycemicIndex":45}}

Example G - Specific branded food:
Input: "2 slices of pepperoni pizza from Domino's"
Output: {"payload":{"description":"2 slices of pepperoni pizza from Domino's","calories":560,"carbsGrams":64,"proteinGram":24,"fatGram":22,"glycemicIndex":70}}

Example H - Generic location, no food (error):
Input: "Restaurant meal"
Output: {"error":"Could you please specify what you ordered at the restaurant?"}

Example I - Generic category only (error):
Input: "Healthy stuff"
Output: {"error":"Could you please be more specific about what healthy foods you ate?"}

Example J - No food info (error):
Input: "Things"
Output: {"error":"Could you please be more specific about what food you consumed?"}
${ragExamples}
SPECIFICITY RULES (Follow strictly):
✅ RETURN PAYLOAD if query contains identifiable food names:
   - Specific foods: "pizza", "pasta", "burger", "sandwich", "salad", "chicken", "steak"
   - Simple items: "nuts", "apple", "eggs", "cheese", "cookies", "quinoa"
   - Beverages: "coconut water", "smoothie", "juice", "milk"
   - Generic brands: "protein bar", "energy bar", "cereal"
   - Branded items: "Domino's pizza", "Subway sandwich"
   → Estimate standard portions if quantity missing

❌ RETURN ERROR if query is too generic:
   - Location only: "restaurant meal", "takeout", "dined in"
   - Generic categories: "healthy stuff", "snack", "food", "things"
   - Context only: "party food", "hospital food"
   → Ask for specific food details

INSTRUCTIONS:
1. FIRST: Check if query contains an IDENTIFIABLE FOOD NAME → Return PAYLOAD (ignore RAG errors)
2. ONLY if NO identifiable food AND only location/category → Return ERROR
3. Ignore Similar Examples unless they have the EXACT or SIMILAR food items
4. For single foods without quantity, estimate standard portions
5. NEVER return both "payload" and "error" fields (or empty error field)
6. Glycemic Index: Low (15-35), Medium (40-60), High (70-85)

Response: Valid JSON only

User query:`;
}

function createHistory(query, top3) {
  const prompt = mealPrompt(responseSchema, top3);
  const history = [
    {
      role: "session",
      content: "reset",
    },
    {
      role: "system",
      content: prompt,
    },
    {
      role: "user",
      content: query,
    },
  ];

  return history;
}









let embeddingModelId;
let llmModelId;
const initEmbeddingModel = async () => {
  embeddingModelId = await loadModel({
    modelSrc: GTE_LARGE_FP16,
    modelType: "embeddings",
    onProgress: (progress) => {
      process.stdout.write(`\rLoading model... ${progress.percentage.toFixed(4)}%`);
    },
  });
};

const initLlmModel = async () => {
  llmModelId = await loadModel({
    modelSrc: QWEN_3_1_7B_INST_Q4,
    modelType: "llm",
    modelConfig: {
      gpu_layers: 999,
      ctx_size: 2048,
      device: "gpu",
    },
    onProgress: (progress) => {
      process.stdout.write(`\rLoading model... ${progress.percentage.toFixed(4)}%`);
    },
  });
};

let db;
const initVectorDb = async () => {
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
  for (const sample of seedData1000plus) {
    const embedding = await embed({ modelId: embeddingModelId, text: sample.prompt });
    db.exec({
      sql: "INSERT INTO documents VALUES (?, ?, ?, vector_as_f32(?))",
      bind: [
        sample.id,
        sample.prompt,
        JSON.stringify(sample.expected_output),
        JSON.stringify(embedding),
      ],
    });
  }

  // Initialize and optimize vector index
  db.exec(
    `SELECT vector_init('documents', 'embedding', 'type=FLOAT32,dimension=1024')`
  );

  // Quantize vectors
  db.exec(`SELECT vector_quantize('documents', 'embedding')`);

  // [Optional] Preload quantized vectors in memory for optimal performance
  db.exec(`SELECT vector_quantize_preload('documents', 'embedding')`);
};

const performVectorSearch = async (query) => {
  console.log("Performing vector search for query:", query);
  const queryEmbedding = await embed({ modelId: embeddingModelId, text: query });
  console.log("Query embedding:", queryEmbedding);
  
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
  console.log("Vector search results:", results);
  return results;
};

const main = async () => {
  await initEmbeddingModel();
  await initVectorDb();

  await initLlmModel();

  const filePath = 'meal/benchmark-results/sqlite-rag-powered/' + new Date().toISOString() + '.json';

  for (const sample of mealDatasetOriginal) {
    const benchmarkResult = {
      prompt: sample.prompt,
      expected_output: sample.expected_output,
    };

    const top3 = await performVectorSearch(sample.prompt);
    benchmarkResult.top3 = top3;

    const history = createHistory(sample.prompt, top3);
    const response = completion({
      modelId: llmModelId, 
      history, 
      stream: true
    });
    let text = "";
    for await (const token of response.tokenStream) {
      process.stdout.write(token);
      text += token;
    }
    benchmarkResult.response = text;

    const stats = await response.stats;
    benchmarkResult.stats = stats;

    try {
      const jsonString = extractJSON(benchmarkResult.response);
      const parsedResult = responseSchema.parse(JSON.parse(jsonString));
      benchmarkResult.actual = parsedResult;

      // Classify the result
      const expectsPayload = !!sample.expected_output.payload;
      const expectsError = !!sample.expected_output.error;
      const hasPayload = !!parsedResult.payload;
      const hasError = !!parsedResult.error;

      if (expectsPayload && hasPayload) {
        benchmarkResult.classification = "truthy_payload";
        benchmarkResult.metrics = calculatePayloadMetrics(
          sample.expected_output.payload,
          parsedResult.payload
        );
      } else if (expectsPayload && !hasPayload) {
        benchmarkResult.classification = "falsy_payload";
      } else if (expectsError && hasError) {
        benchmarkResult.classification = "truthy_error";
      } else if (expectsError && !hasError) {
        benchmarkResult.classification = "falsy_error";
      } else if (!expectsPayload && hasPayload) {
        benchmarkResult.classification = "falsy_payload";
      } else if (!expectsError && hasError) {
        benchmarkResult.classification = "falsy_error";
      }
    } catch (error) {
      benchmarkResult.parseError =
        error instanceof Error ? error.message : String(error);

      // If we expect an error and got a parse error, it might be a falsy error
      if (sample.expected_output.error) {
        benchmarkResult.classification = "falsy_error";
      } else {
        benchmarkResult.classification = "falsy_payload";
      }
    }
    await writeResultIncrementally(benchmarkResult, filePath);
  }

  process.kill(process.pid);
};

main().catch(console.error);

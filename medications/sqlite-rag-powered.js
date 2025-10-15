import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import sqlite3InitModule from "@sqliteai/sqlite-wasm";
import comprehensiveSeedData from "./medications-datasets/comprehensive-seed-data.json" with { type: "json" };
import medicationsTestDataset from "./medications-datasets/test-dataset.json" with { type: "json" };
import { extractJSON, writeResultIncrementallyMedications, compareMedicationPayloads } from "../utils.js";

const IntervalUnitEnum = ["minutes", "hours", "days"]
const MedicationUnitEnum = [
  "mg", // milligrams
  "g", // grams
  "ml", // milliliters
  "mcg", // micrograms
  "IU", // international units
  "tablet", // tablets/pills
  "capsule", // capsules
  "drop", // drops
  "spray", // sprays
  "patch", // patches
  "injection", // injections
]

const MedicationFrequencyEnum = [
  "once", // one-time medication
  "interval", // custom interval (every X minutes/hours/days)
  "daily", // every day
  "weekly", // every week
  "monthly", // every month
  "as_needed", // PRN (as needed)
]

const medicationEventSchema = z.object({
  name: z.string().min(1, "Medication name is required"),
  dosage: z.number().positive("Dosage must be positive"),
  unit: z.enum(MedicationUnitEnum),
  notes: z.string().optional(),
  // For reminder medications
  frequency: z.enum(MedicationFrequencyEnum).optional(),
  reminderTime: z.string().optional(), // HH:MM format for daily reminders
  reminderDays: z.array(z.number().min(0).max(6)).optional(), // 0-6 for Sun-Sat
  intervalValue: z.number().optional(), // For interval frequency: every X
  intervalUnit: z.enum(IntervalUnitEnum).optional(), // minutes, hours, or days
  isReminder: z.boolean().default(false),
  // Tracking
  taken: z.boolean().default(true), // false for missed doses
});

const responseSchema = z.object({
  error: z.string().optional(),
  payload: medicationEventSchema.optional(),
});

function medicationPrompt(schema, top3) {
  // Build RAG examples section
  let ragExamples = "";
  if (top3 && top3.length > 0) {
    ragExamples = "\n--- SIMILAR EXAMPLES FROM DATABASE ---\n";
    top3.forEach((example, idx) => {
      ragExamples += `\nExample ${idx + 1} (similarity: ${example.distance?.toFixed(1) || 'N/A'}):\nInput: "${example.prompt}"\nOutput: ${JSON.stringify(example.expected_output)}\n`;
    });
    ragExamples += "\n⚠️ If query closely matches these examples, follow their structure. Otherwise use baseline rules below.\n";
  }

  return `/no_think
Parse medication query and output JSON with EITHER "payload" OR "error" field (never both, never empty strings).

Schema:
${JSON.stringify(z.toJSONSchema(schema), null, 2)}
${ragExamples}
--- PARSING RULES ---

UNITS: mg, g, ml, mcg, IU, tablet, capsule, drop, spray, patch, injection

FREQUENCY:
• Past tense ("took", "had", "popped", "downed") → "once"
• "daily"/"every day" → "daily"
• "weekly"/"every week" → "weekly"
• "monthly"/"every month" → "monthly"
• "as needed"/"when needed"/"PRN" → "as_needed"
• "every X minutes/hours/days" → "interval" (set intervalValue & intervalUnit)

TAKEN FIELD (critical):
• "took", "had", "used", "applied", "injected" → taken=true
• "forgot", "missed", "didn't take", "haven't taken" → taken=false
• "need to", "remind", "alert", "every X" → taken=false

REMINDER DETECTION:
• "remind", "notify", "alert", "tell me", "remember", "don't forget" → isReminder=true
• "need to take X every Y" → isReminder=true
• "take X every Y" (future context) → isReminder=true
• Past tense only → isReminder=false

EXAMPLES:

A. Interval reminder:
"Take 500mg paracetamol every 6 hours, notify me"
→ {"payload":{"name":"Paracetamol","dosage":500,"unit":"mg","frequency":"interval","intervalValue":6,"intervalUnit":"hours","isReminder":true,"taken":false}}

B. Past dose:
"I took 500mg aspirin"
→ {"payload":{"name":"Aspirin","dosage":500,"unit":"mg","frequency":"once","isReminder":false,"taken":true}}

C. Missed dose:
"Didn't take my 50mg sertraline today"
→ {"payload":{"name":"sertraline","dosage":50,"unit":"mg","frequency":"once","isReminder":false,"taken":false}}

D. Casual language:
"popped some aspirin"
→ {"payload":{"name":"Aspirin","dosage":1,"unit":"tablet","frequency":"once","isReminder":false,"taken":true}}

E. Generic medication (accept):
"Missed my evening dose of blood pressure medication"
→ {"payload":{"name":"blood pressure medication","dosage":1,"unit":"tablet","frequency":"once","isReminder":false,"taken":false,"notes":"missed evening dose"}}

F. Too vague (error):
"Remind me about vitamins"
→ {"error":"Which vitamin and when should I remind you?"}

IDENTIFIABLE MEDICATIONS (accept these):
• Specific: aspirin, ibuprofen, vitamin D, melatonin, etc.
• Generic but clear: "blood pressure medication", "thyroid medication", "pain medication"
• Branded: Tylenol, Advil, Tums, etc.

TOO VAGUE (error):
• "medicine", "medication", "pills", "meds", "vitamins" (without type)
• Missing both medication name AND context

CRITICAL:
1. NEVER output both "payload" and "error"
2. NEVER output empty strings: "", "none", "false" in error field
3. If returning error, ONLY set "error" field (omit "payload" completely)
4. If returning payload, ONLY set "payload" field (omit "error" completely)
5. Misspellings: "aspirine"→"Aspirin", "ibuprofin"→"Ibuprofen"
6. Defaults: dosage=1, unit="tablet" if missing
7. reminderTime: ONLY for clock times ("at 9am"→"09:00"), NOT intervals
8. "every X minutes/hours/days" MUST use frequency="interval" + intervalValue + intervalUnit

User query:`;
}

function createHistory(query, top3) {
  const prompt = medicationPrompt(responseSchema, top3);
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
      process.stdout.write(`\rLoading embedding model... ${progress.percentage.toFixed(4)}%`);
    },
  });
  console.log("\n✅ Embedding model loaded");
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
      process.stdout.write(`\rLoading LLM model... ${progress.percentage.toFixed(4)}%`);
    },
  });
  console.log("\n✅ LLM model loaded");
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

  console.log(`\n📚 Embedding ${comprehensiveSeedData.length} medication documents...`);
  let count = 0;
  for (const sample of comprehensiveSeedData) {
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
    count++;
    if (count % 100 === 0) {
      process.stdout.write(`\rEmbedded ${count}/${comprehensiveSeedData.length} documents...`);
    }
  }
  console.log(`\n✅ All documents embedded`);

  // Initialize and optimize vector index
  console.log("🔧 Initializing vector index...");
  db.exec(
    `SELECT vector_init('documents', 'embedding', 'type=FLOAT32,dimension=1024')`
  );

  // Quantize vectors
  console.log("🔧 Quantizing vectors...");
  db.exec(`SELECT vector_quantize('documents', 'embedding')`);

  // [Optional] Preload quantized vectors in memory for optimal performance
  console.log("🔧 Preloading quantized vectors...");
  db.exec(`SELECT vector_quantize_preload('documents', 'embedding')`);
  console.log("✅ Vector database ready");
};

const performVectorSearch = async (query) => {
  const queryEmbedding = await embed({ modelId: embeddingModelId, text: query });
  
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

const main = async () => {
  console.log("🚀 Starting RAG-powered medication benchmark...\n");
  
  await initEmbeddingModel();
  await initVectorDb();
  await initLlmModel();

  const filePath = 'medications/benchmark-results/sqlite-rag-powered/' + new Date().toISOString() + '.json';

  console.log(`\n📊 Running benchmark on ${medicationsTestDataset.length} test cases...\n`);

  for (const sample of medicationsTestDataset) {
    console.log(`\n${"=".repeat(80)}`);
    console.log(`Processing: "${sample.prompt}"`);
    
    const benchmarkResult = {
      prompt: sample.prompt,
      expected_output: sample.expected_output,
    };

    // Perform RAG search
    const top3 = await performVectorSearch(sample.prompt);
    benchmarkResult.top3 = top3;
    console.log(`\nTop 3 similar examples:`);
    top3.forEach((ex, idx) => {
      console.log(`  ${idx + 1}. [${ex.distance.toFixed(2)}] "${ex.prompt}"`);
    });

    const history = createHistory(sample.prompt, top3);
    const response = completion({
      modelId: llmModelId,
      history,
      stream: true,
    });
    
    console.log(`\nModel response:`);
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
        // Deep comparison for truthy payloads (notes excluded from scoring)
        const comparison = compareMedicationPayloads(
          sample.expected_output.payload,
          parsedResult.payload
        );
        benchmarkResult.payloadComparison = comparison;
        
        // Define threshold: 80% or higher = success
        const MATCH_THRESHOLD = 80;
        
        if (comparison.matchPercentage >= MATCH_THRESHOLD) {
          benchmarkResult.classification = "truthy_payload";
          console.log(`\n✅ TRUTHY_PAYLOAD (${comparison.matchPercentage.toFixed(1)}% match)`);
        } else if (comparison.matchPercentage > 0) {
          // Some fields match but not enough
          benchmarkResult.classification = "partial_payload";
          console.log(`\n⚠️  PARTIAL_PAYLOAD (${comparison.matchPercentage.toFixed(1)}% match)`);
        } else {
          // No fields match at all
          benchmarkResult.classification = "falsy_payload";
          console.log(`\n❌ FALSY_PAYLOAD (0% match)`);
        }
      } else if (expectsPayload && !hasPayload) {
        benchmarkResult.classification = "falsy_payload";
        console.log(`\n❌ FALSY_PAYLOAD (expected payload, got error)`);
      } else if (expectsError && hasError) {
        benchmarkResult.classification = "truthy_error";
        console.log(`\n✅ TRUTHY_ERROR`);
      } else if (expectsError && !hasError) {
        benchmarkResult.classification = "falsy_error";
        console.log(`\n❌ FALSY_ERROR (expected error, got payload)`);
      } else if (!expectsPayload && hasPayload) {
        benchmarkResult.classification = "falsy_payload";
        console.log(`\n❌ FALSY_PAYLOAD (unexpected payload)`);
      } else if (!expectsError && hasError) {
        benchmarkResult.classification = "falsy_error";
        console.log(`\n❌ FALSY_ERROR (unexpected error)`);
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
      console.log(`\n❌ PARSE ERROR: ${benchmarkResult.parseError.substring(0, 100)}`);
    }
    
    await writeResultIncrementallyMedications(benchmarkResult, filePath);
  }

  console.log(`\n${"=".repeat(80)}`);
  console.log(`✅ Benchmark complete! Results saved to: ${filePath}`);
  process.kill(process.pid);
};

main().catch(console.error);


import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import comprehensiveSeedDataWithEmbeddings from "./medications-datasets/comprehensive-seed-data-with-embeddings.json" with { type: "json" };
import medicationsTestDataset from "./medications-datasets/test-dataset.json" with { type: "json" };
import { extractJSON, writeResultIncrementallyMedications, compareMedicationPayloads } from "../utils.js";

const IntervalUnitEnum = ["minutes", "hours", "days"];
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
];

const MedicationFrequencyEnum = [
  "once", // one-time medication
  "interval", // custom interval (every X minutes/hours/days)
  "daily", // every day
  "weekly", // every week
  "monthly", // every month
  "as_needed", // PRN (as needed)
];

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
    ragExamples = "\n--- SIMILAR EXAMPLES ---\n";
    top3.forEach((example, idx) => {
      const sim = example.similarity ? (example.similarity * 100).toFixed(2) : 'N/A';
      ragExamples += `${idx + 1}. [${sim}%] "${example.prompt}" → ${JSON.stringify(example.expected_output)}\n`;
    });
  }

  return `/no_think
Parse medication queries to JSON.

Schema: ${JSON.stringify(z.toJSONSchema(schema))}

⚠️ CRITICAL DECISION RULE (HIGHEST PRIORITY):
Step 1: Does query mention specific medication/drug name?
Step 2: YES → RETURN PAYLOAD (extract dosage, unit, frequency, taken status)
Step 3: NO → RETURN ERROR

Core Examples:
"Took 500mg aspirin" → {"payload":{"name":"Aspirin","dosage":500,"unit":"mg","frequency":"once","isReminder":false,"taken":true}}
"Popped some aspirin" → {"payload":{"name":"Aspirin","dosage":1,"unit":"tablet","frequency":"once","isReminder":false,"taken":true}}
"Didn't take my 50mg sertraline" → {"payload":{"name":"Sertraline","dosage":50,"unit":"mg","frequency":"once","isReminder":false,"taken":false}}
"Missed my blood pressure med" → {"payload":{"name":"blood pressure medication","dosage":1,"unit":"tablet","frequency":"once","isReminder":false,"taken":false}}
"Take 500mg paracetamol every 6 hours" → {"payload":{"name":"Paracetamol","dosage":500,"unit":"mg","frequency":"interval","intervalValue":6,"intervalUnit":"hours","isReminder":true,"taken":false}}
"Remind me about vitamins" (no specific vitamin) → {"error":"Which vitamin and when?"}
"Pills" (no specific medication) → {"error":"Which medication?"}

WRONG Examples (do NOT follow):
❌ "Took aspirin" → error (aspirin IS a medication!)
❌ "Missed my blood pressure med" → error (blood pressure medication IS identifiable!)
❌ "Popped some Advil" → error (Advil IS a medication!)
${ragExamples}
Use Similar Examples for dosage/unit patterns and frequency extraction.
IGNORE Similar Examples that show errors when query mentions specific medication.

FREQUENCY RULES:
• Past tense ("took", "had", "popped") → "once"
• "daily"/"every day" → "daily"
• "weekly"/"every week" → "weekly"
• "monthly"/"every month" → "monthly"
• "as needed"/"PRN" → "as_needed"
• "every X minutes/hours/days" → "interval" (set intervalValue & intervalUnit)

TAKEN STATUS:
• "took", "had", "used", "applied", "injected" → taken=true
• "forgot", "missed", "didn't take", "haven't taken" → taken=false
• "remind", "need to", "alert", "every X" (future) → taken=false

REMINDER DETECTION:
• "remind", "notify", "alert", "tell me", "remember" → isReminder=true
• "need to take X every Y" → isReminder=true
• Past tense only → isReminder=false

IDENTIFIABLE MEDICATIONS (accept):
• Specific: aspirin, ibuprofen, vitamin D, melatonin, metformin, lisinopril
• Generic but clear: "blood pressure medication", "thyroid medication", "pain medication"
• Branded: Tylenol, Advil, Tums, Zyrtec

TOO VAGUE (error):
• "medicine", "medication", "pills", "meds", "vitamins" (without type)
• Missing both medication name AND context

DEFAULTS:
• dosage=1, unit="tablet" if missing
• Misspellings: "aspirine"→"Aspirin", "ibuprofin"→"Ibuprofen"

RULES:
• NEVER return both "payload" and "error"
• NEVER return empty error strings
• reminderTime: ONLY for clock times ("at 9am"→"09:00"), NOT intervals
• "every X minutes/hours/days" MUST use frequency="interval" + intervalValue + intervalUnit

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
};

function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have the same length");
  }
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magnitudeA += vecA[i] * vecA[i];
    magnitudeB += vecB[i] * vecB[i];
  }
  const magnitude = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
  if (magnitude === 0) {
    return 0;
  }
  return dotProduct / magnitude;
}

function getTop3Samples(queryEmbedding) {
  const samplesWithSimilarity = comprehensiveSeedDataWithEmbeddings.map((sample) => ({
    ...sample,
    similarity: cosineSimilarity(queryEmbedding, sample.embedding),
  }));
  samplesWithSimilarity.sort((a, b) => b.similarity - a.similarity);
  return samplesWithSimilarity.slice(0, 3);
}

const main = async () => {
  await initEmbeddingModel();
  await initLlmModel();

  const filePath = 'medications/benchmark-results/rag-over-mem/' + new Date().toISOString() + '.json';

  for (const sample of medicationsTestDataset) {
    const benchmarkResult = {
      prompt: sample.prompt,
      expected_output: sample.expected_output,
    };

    const queryEmbedding = await embed({ modelId: embeddingModelId, text: sample.prompt });
    let top3 = getTop3Samples(queryEmbedding);
    top3 = top3.map((s) => {
      // eslint-disable-next-line no-unused-vars
      const { embedding, ...rest } = s;
      return rest;
    });
    benchmarkResult.top3 = top3;

    const history = createHistory(sample.prompt, top3);
    const response = completion({
      modelId: llmModelId,
      history,
      stream: true,
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
        } else if (comparison.matchPercentage > 0) {
          // Some fields match but not enough
          benchmarkResult.classification = "partial_payload";
        } else {
          // No fields match at all
          benchmarkResult.classification = "falsy_payload";
        }
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
    
    await writeResultIncrementallyMedications(benchmarkResult, filePath);
  }

  process.kill(process.pid);
};

main().catch(console.error);


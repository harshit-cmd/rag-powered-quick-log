import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
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

function medicationPrompt(schema) {
  return `/no_think
    You are given a schema for a medication tool call and you need to fill it based on the user query. Here's the schema:

    ${JSON.stringify(z.toJSONSchema(schema))}

    GOOD examples (specific with details):
    - "I took 500mg aspirin for my headache" ✓
    - "Remind me to take vitamin D every day at 9am" ✓
    - "Need to take 2 tablets of vitamin D every morning at 8am" ✓
    - "Just took my 10mg melatonin before bed" ✓
    - "Missed my evening dose of blood pressure medication" ✓
    - "I need to remember to take aspirin daily at 9am" ✓
    - "Take 500mg paracetamol every 6 hours, notify me" ✓
    - "Take aspirin every 2 hours, alert me" ✓
    - "Ibuprofen 400mg every 4 hours, remind me" ✓

    BAD examples (but still estimate):
    - "I took aspirin" → Estimate: 1 tablet, frequency="once", isReminder=false, taken=true
    - "Remind me about vitamins" → ERROR: Which vitamin and when?
    - "Need vitamins" → ERROR: Which vitamin and when?
    - "Took medicine" → ERROR: What specific medication?
    
    COMPLEX PARSING EXAMPLES:
    - "Take 500mg paracetamol every 6 hours, notify me" → name="Paracetamol", dosage=500, unit="mg", frequency="interval", intervalValue=6, intervalUnit="hours", isReminder=true, taken=false
    - "Aspirin every 2 minutes" → name="Aspirin", dosage=1, unit="tablet", frequency="interval", intervalValue=2, intervalUnit="minutes", isReminder=true, taken=false
    - "400mg ibuprofen every 4 hours alert me" → name="Ibuprofen", dosage=400, unit="mg", frequency="interval", intervalValue=4, intervalUnit="hours", isReminder=true, taken=false

    MEDICATION UNITS (use exactly these):
    mg, g, ml, mcg, IU, tablet, capsule, drop, spray, patch, injection

    FREQUENCY MAPPING:
    - "I took", "just took", past tense → "once"
    - "every day", "daily" → "daily" 
    - "weekly", "every week" → "weekly"
    - "monthly", "every month" → "monthly"
    - "as needed", "when needed" → "as_needed"
    - "every X minutes/hours/days" → "interval" (set intervalValue=X, intervalUnit="minutes"/"hours"/"days")

    INTERVAL FREQUENCY EXAMPLES:
    - "every 2 minutes" → frequency="interval", intervalValue=2, intervalUnit="minutes"
    - "every 6 hours" → frequency="interval", intervalValue=6, intervalUnit="hours"
    - "every 3 days" → frequency="interval", intervalValue=3, intervalUnit="days"
    - "every 30 minutes" → frequency="interval", intervalValue=30, intervalUnit="minutes"

    REMINDER DETECTION:
    - "remind me", "notify me", "alert me", "tell me", "remember to" → isReminder=true
    - "need to take...every", "take...every" with any interval → isReminder=true
    - "take X, notify/remind/alert me" → isReminder=true
    - Past tense ("took", "had", "just took") → isReminder=false, taken=true
    - Future/reminder language with any interval (every X minutes/hours/days) → isReminder=true

    RULES:
    - Always include medication name (fix common misspellings: "aspirine"→"Aspirin", "paracetamol"→"Paracetamol")
    - Default dosage to 1 if not specified  
    - Default unit to "tablet" if not specified
    - Set frequency based on context (see mapping above)
    - For ANY "every X minutes/hours/days" → MUST set frequency="interval", intervalValue=X, intervalUnit="minutes"/"hours"/"days"
    - If query contains "notify me", "remind me", "alert me", "tell me" → isReminder=true
    - If query says "take X every Y" (any interval) → isReminder=true, taken=false
    - Set "taken" to true for completed doses, false for reminders/missed doses
    - Include "reminderTime" ONLY when specific clock time mentioned (e.g., "at 9am"→"09:00")
    - For interval frequency (every X minutes/hours/days), do NOT set reminderTime
    - For daily/weekly/monthly WITHOUT specific time, default reminderTime="09:00"
    - Use "notes" for additional context (missed dose, with food, etc.)

    CRITICAL:
    - Set ONLY "payload" field if you can create the log (even with estimates)
    - Set ONLY "error" field if medication is too vague to identify
    - Use error if query is completely unrelated to medications
    - Never set both fields

    Output valid JSON only.

    User query:
    `;
}


function createHistory(query) {
  const prompt = medicationPrompt(responseSchema);
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









let llmModelId;

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

const main = async () => {
  await initLlmModel();
  const filePath = 'medications/benchmark-results/current/' + new Date().toISOString() + '.json';

  for (const sample of medicationsTestDataset) {
    const benchmarkResult = {
      prompt: sample.prompt,
      expected_output: sample.expected_output,
    };

    const history = createHistory(sample.prompt);
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
};

main()
  .catch(console.error)
  .finally(() => {
    process.kill(process.pid);
  });


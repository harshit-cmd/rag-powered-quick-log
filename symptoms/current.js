import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import { compareSymptomPayloads, extractJSON, writeResultIncrementallySymptoms } from "../utils.js";
import symptomTestDataset from "./symptoms-datasets/test-dataset.json" with { type: "json" };

export const symptomPayloadSchema = z.object({
  name: z.string().describe("Brief name or category for the symptom"),
  description: z
    .string()
    .describe("Natural language description of what the user is feeling"),
  severity: z
    .enum(["mild", "moderate", "severe"])
    .optional()
    .describe("Severity level if mentioned"),
});

const responseSchema = z.object({
  payload: symptomPayloadSchema.optional(),
  error: z.string().optional(),
});

function symptomPrompt(schema) {
  return `/no_think
    You are given a schema for a symptom tool call and you need to fill it based on the user query. Here's the schema:

    ${JSON.stringify(z.toJSONSchema(schema))}

    GOOD examples (specific with details):
    - "Mild headache on the left side of my head" ✓
    - "Severe nausea and stomach pain after eating" ✓
    - "My knee is aching, probably from yesterday's run" ✓
    - "Feeling very anxious and stressed about work presentation" ✓

    BAD examples (too vague, estimate anyway):
    - "Not feeling well" → Estimate: name="general malaise", description="not feeling well"  
    - "My head hurts" → Estimate: name="headache", severity="moderate"
    - "I'm feeling bad" → ERROR: Please describe what specific symptoms you're experiencing

    RULES:
    - Need symptom name (headache, fatigue, nausea, pain, etc.)
    - Can estimate severity if not specified (mild/moderate/severe)
    - If symptoms too generic ("sick", "bad"), use error field
    - Description captures user's natural language

    Common names: headache, fatigue, nausea, pain, anxiety, dizziness, fever.

    CRITICAL:
    - Set ONLY "payload" field if you can create the log (even with estimates)
    - Set ONLY "error" field if symptoms are too vague to identify
    - Don't just say "too vague", come up with a proper error message
    - ONLY if the query is completely unrelated to symptoms/feelings (like "I went running"), set error: "You want to log a symptom but the query is not about how you're feeling"
    - Never set both fields

    Output valid JSON only.

    User query:
    `;
}


function createHistory(query) {
  const prompt = symptomPrompt(responseSchema);
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
  console.log("🚀 Starting symptoms benchmark...\n");
  await initLlmModel();
  console.log("\n✅ LLM model loaded");
  
  const filePath = 'symptoms/benchmark-results/current/' + new Date().toISOString() + '.json';

  console.log(`\n📊 Running benchmark on ${symptomTestDataset.length} test cases...\n`);

  for (const sample of symptomTestDataset) {
    console.log(`\n${"=".repeat(80)}`);
    console.log(`Processing: "${sample.prompt}"`);
    
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
        // Deep comparison for truthy payloads
        const comparison = compareSymptomPayloads(
          sample.expected_output.payload,
          parsedResult.payload
        );
        benchmarkResult.symptomComparison = comparison;
        
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
    await writeResultIncrementallySymptoms(benchmarkResult, filePath);
  }

  console.log(`\n${"=".repeat(80)}`);
  console.log(`✅ Benchmark complete! Results saved to: ${filePath}`);
};

main()
  .catch(console.error)
  .finally(() => {
    process.kill(process.pid);
  });


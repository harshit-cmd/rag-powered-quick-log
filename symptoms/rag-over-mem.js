import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import comprehensiveSeedDataWithEmbeddings from "./symptoms-datasets/comprehensive-seed-data-with-embeddings.json" with { type: "json" };
import symptomsTestDataset from "./symptoms-datasets/test-dataset.json" with { type: "json" };
import { extractJSON, writeResultIncrementallySymptoms, compareSymptomPayloads } from "../utils.js";

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
  error: z.string().optional(),
  payload: symptomPayloadSchema.optional(),
});

function symptomPrompt(schema, top3) {
  // Build RAG examples section - only show severity and description patterns
  let ragExamples = "";
  if (top3 && top3.length > 0) {
    ragExamples = "\nSimilar cases:\n";
    top3.forEach((example, idx) => {
      if (example.expected_output.payload) {
        const p = example.expected_output.payload;
        ragExamples += `${idx + 1}. "${example.prompt}" → severity:${p.severity || 'moderate'}\n`;
      }
    });
  }

  return `/no_think
Parse to JSON. Schema: ${JSON.stringify(z.toJSONSchema(schema))}

ACCEPT if query mentions ANY symptom/feeling/body issue → {"payload":{...}}
REJECT only if completely unrelated to health → {"error":"..."}

Examples:
"Headache" → {"payload":{"name":"headache","description":"Headache","severity":"moderate"}}
"Dizzy" → {"payload":{"name":"dizziness","description":"Dizzy","severity":"moderate"}}
"Severe nausea" → {"payload":{"name":"nausea","description":"Severe nausea","severity":"severe"}}
"Knee pain" → {"payload":{"name":"knee pain","description":"Knee pain","severity":"moderate"}}
"Anxious" → {"payload":{"name":"anxiety","description":"Anxious","severity":"moderate"}}
"I went running" → {"error":"Query is not about symptoms"}
${ragExamples}
Severity from query:
- mild: slight/minor/little
- moderate: default if not specified
- severe: severe/intense/bad/terrible/extreme

CRITICAL - Name field (most important):
- Extract primary symptom from query
- For pain: include location (e.g., "knee pain", "back pain")
- For multiple symptoms: pick main one or combine (e.g., "nausea and pain")
- Single word OK if clear (e.g., "headache", "nausea", "dizziness")

Description: brief summary of what user is feeling

Valid JSON only. Never both payload & error.

User query:`;
}

function createHistory(query, top3) {
  const prompt = symptomPrompt(responseSchema, top3);
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

  const filePath = 'symptoms/benchmark-results/rag-over-mem/' + new Date().toISOString() + '.json';

  for (const sample of symptomsTestDataset) {
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
        // Simple comparison: only check if name is semantically correct
        const comparison = compareSymptomPayloads(
          sample.expected_output.payload,
          parsedResult.payload
        );
        benchmarkResult.symptomComparison = comparison;
        
        // Simple: either it matches or it doesn't
        benchmarkResult.classification = comparison.isMatch 
          ? "truthy_payload" 
          : "falsy_payload";
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
    
    await writeResultIncrementallySymptoms(benchmarkResult, filePath);
  }

  process.kill(process.pid);
};

main().catch(console.error);


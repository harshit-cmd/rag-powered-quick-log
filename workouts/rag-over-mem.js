import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import workoutRagDatasetWithEmbeddings from "./workouts-datasets/workout-rag-dataset-cleaned-with-embeddings.json" with { type: "json" };
import workoutTestDataset from "./workouts-datasets/workout-test-dataset.json" with { type: "json" };
import { extractJSON, writeResultIncrementallyWorkouts, calculateWorkoutPayloadMetrics } from "../utils.js";

export const workoutPayloadSchema = z.object({
  workoutType: z.string().optional(),
  description: z.string().optional(),
  durationMinutes: z.number().optional(),
  caloriesBurned: z.number().optional(),
  intensityLevel: z.string().optional(),
  exercises: z
    .array(
      z.object({
        name: z.string(),
        sets: z.number().optional(),
        reps: z.number().optional(),
        weight: z.number().optional(),
        weightUnit: z.string().optional(),
        durationMinutes: z.number().optional(),
      }),
    )
    .optional(),
});

const responseSchema = z.object({
  payload: workoutPayloadSchema.optional(),
  error: z.string().optional(),
});

function workoutPrompt(schema, top3) {
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
Parse workout queries to JSON. Schema: ${JSON.stringify(z.toJSONSchema(schema))}

⚠️ RULE: Specific activity (run/swim/yoga/bench press/squats) → PAYLOAD | Only vague (gym/cardio/workout/stairs/sports) → ERROR

Examples:
"Ran 30min" → {"payload":{"workoutType":"running","description":"30 min run","durationMinutes":30,"caloriesBurned":300,"intensityLevel":"moderate"}}
"Bench 3x10 50kg" → {"payload":{"workoutType":"strength training","description":"Bench press","durationMinutes":15,"caloriesBurned":90,"exercises":[{"name":"bench press","sets":3,"reps":10,"weight":50,"weightUnit":"kg"}]}}
"Gym" → {"error":"What workout did you do?"}
${ragExamples}
Use examples if they follow the RULE.

Types: running|walking|cycling|swimming|rowing|elliptical|strength training|weightlifting|calisthenics|yoga|pilates|stretching|HIIT|circuit training|CrossFit|basketball|soccer|tennis

Duration: Use given OR distance÷speed OR estimate (cardio:30, strength:45, yoga:45, HIIT:25)

Calories = min × rate: Run:10 Walk:5 Cycle:8 Swim:11 Strength:6 Yoga:3 HIIT:12
Examples: 30min run=30×10=300 | 45min strength=45×6=270 | Adjust ±20% intensity

Intensity: high(intense/hard/fast) | moderate(default) | low(easy/light/gentle)

Exercises (strength only): Sets=# sets NOT reps! | "50 pushups"=sets:5 reps:10 | Units: kg/lbs

CRITICAL: REJECT vague terms alone | Calculate calories correctly | Never both payload & error | Valid JSON

User query:`;
}

function createHistory(query, top3) {
  const prompt = workoutPrompt(responseSchema, top3);
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
  const samplesWithSimilarity = workoutRagDatasetWithEmbeddings.map((sample) => ({
    ...sample,
    similarity: cosineSimilarity(queryEmbedding, sample.embedding),
  }));
  samplesWithSimilarity.sort((a, b) => b.similarity - a.similarity);
  return samplesWithSimilarity.slice(0, 3);
}

const main = async () => {
  await initEmbeddingModel();
  await initLlmModel();

  const filePath = 'workouts/benchmark-results/rag-over-mem/' + new Date().toISOString() + '.json';

  for (const sample of workoutTestDataset) {
    const benchmarkResult = {
      id: sample.id,
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
        benchmarkResult.classification = "truthy_payload";
        benchmarkResult.metrics = calculateWorkoutPayloadMetrics(
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
    
    await writeResultIncrementallyWorkouts(benchmarkResult, filePath);
  }

  process.kill(process.pid);
};

main().catch(console.error);


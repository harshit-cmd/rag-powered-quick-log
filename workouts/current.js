import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import workoutDataset from "./workouts-datasets/workout-test-dataset.json" with { type: "json" };
import { calculateWorkoutPayloadMetrics, extractJSON, writeResultIncrementallyWorkouts } from "../utils.js";

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

export function workoutPrompt(schema) {
  return `/no_think
    You are given a schema for a workout tool call and you need to fill it based on the user query. Here's the schema:

    ${JSON.stringify(z.toJSONSchema(schema))}

    GOOD examples (specific with details):
    - "I ran for 30 minutes and burned 300 calories" ✓
    - "3 sets of 10 reps bench press with 50kg" ✓
    - "45 minute yoga session, mostly vinyasa flow" ✓
    - "Swimming laps for 25 minutes, burned about 200 calories" ✓
    - "High intensity interval training for 20 minutes" ✓

    BAD examples (but still estimate):
    - "I went for a run" → Estimate: 30 min running, 300 calories
    - "Did some pushups" → Estimate: 3 sets of 10 reps, 50 calories
    - "I ran" → Estimate: 30 min running, 300 calories
    - "Worked out" → ERROR: What type of workout?

    CALORIE ESTIMATION GUIDELINES:
    - Running: ~10 cal/min (300 cal for 30 min)
    - Walking: ~5 cal/min (150 cal for 30 min)
    - Cycling: ~8 cal/min (240 cal for 30 min)
    - Swimming: ~11 cal/min (275 cal for 25 min)
    - Strength training: ~6 cal/min (180 cal for 30 min)
    - Yoga: ~3 cal/min (90 cal for 30 min)
    - HIIT: ~12 cal/min (240 cal for 20 min)
    - Adjust based on intensity mentioned (high/low/moderate)

    RULES:
    - Always include workoutType (running, cycling, strength training, etc.)
    - Always include a brief description of what they did
    - Estimate durationMinutes if not specified (20-45 min typical)
    - ALWAYS estimate caloriesBurned based on activity and duration
    - Include intensityLevel if mentioned (low, moderate, high)
    - For strength training, include exercises array with sets/reps/weight
    - Only error if completely unclear ("exercised") or unrelated to fitness

    CRITICAL:
    - Set ONLY "payload" field if you can create the log (always estimate calories)
    - Set ONLY "error" field if workout is too vague to identify
    - Use error if the query is completely unrelated to exercise/fitness
    - Never set both fields

    Output valid JSON only.

    User query:
    `;
}



function createHistory(query) {
  const prompt = workoutPrompt(responseSchema);
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
  const filePath = 'workouts/benchmark-results/current/' + new Date().toISOString() + '.json';

  for (const sample of workoutDataset) {
    const benchmarkResult = {
      id: sample.id,
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
};

main()
  .catch(console.error)
  .finally(() => {
    process.kill(process.pid);
  });


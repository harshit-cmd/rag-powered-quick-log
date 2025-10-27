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
Parse workout queries to JSON. Schema: ${JSON.stringify(z.toJSONSchema(schema))}

⚠️ RULE: Specific activity (run/swim/yoga/bench press/squats) → PAYLOAD | Only vague (gym/cardio/workout/stairs/sports) → ERROR

Examples:
"Ran 30min" → {"payload":{"workoutType":"running","description":"30 min run","durationMinutes":30,"caloriesBurned":300,"intensityLevel":"moderate"}}
"Bench 3x10 50kg" → {"payload":{"workoutType":"strength training","description":"Bench press","durationMinutes":15,"caloriesBurned":90,"exercises":[{"name":"bench press","sets":3,"reps":10,"weight":50,"weightUnit":"kg"}]}}
"50 pushups" → {"payload":{"workoutType":"strength training","description":"Pushups","durationMinutes":10,"caloriesBurned":60,"exercises":[{"name":"pushups","sets":5,"reps":10}]}}
"Gym" → {"error":"What workout did you do?"}
"Cardio" → {"error":"What type of cardio?"}

Types: running|walking|cycling|swimming|rowing|elliptical|strength training|weightlifting|calisthenics|yoga|pilates|stretching|HIIT|circuit training|CrossFit|basketball|soccer|tennis

Duration: Use given OR distance÷speed OR estimate (cardio:30, strength:45, yoga:45, HIIT:25)

Calories = min × rate: Run:10 Walk:5 Cycle:8 Swim:11 Strength:6 Yoga:3 HIIT:12
Examples: 30min run=30×10=300 | 45min strength=45×6=270 | Adjust ±20% intensity

Intensity: high(intense/hard/fast) | moderate(default) | low(easy/light/gentle)

Exercises (strength only): Sets=# sets NOT reps! | "50 pushups"=sets:5 reps:10 | Units: kg/lbs

CRITICAL: REJECT vague terms alone | Calculate calories correctly | Never both payload & error | Valid JSON

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


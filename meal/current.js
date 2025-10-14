import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
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

export function mealPrompt(schema) {
  return `/no_think
    You are given a schema for a meal tool call and you need to fill it based on the user query. Here's the schema:

    ${JSON.stringify(z.toJSONSchema(schema))}

    GOOD examples (specific with quantities):
    - "I had a 250g steak with 100g mixed roasted vegetables" ✓
    - "Ate 1 cup oatmeal with 1 medium banana" ✓  
    - "Had 1 hot dog with mustard" ✓ (hot dog is a known item: bun + sausage)
    - "2 slices whole grain toast with 2 eggs" ✓

    SIMPLE SNACKS/SINGLE ITEMS (always acceptable, estimate standard portions):
    - "I ate banana" → 1 medium banana (~120g)
    - "I ate apple" → 1 medium apple (~180g)
    - "I drink 100mg espresso" → 1 shot espresso with 100mg caffeine
    - "Had nuts" → handful of mixed nuts (~30g)
    - "Ate chocolate" → 1 piece/square dark chocolate (~20g)

    ESTIMATE when quantities missing but food is identifiable:
    - "I had steak and veggies" → Estimate: 200g steak, 80g vegetables
    - "Ate pasta" → Estimate: 1 cup cooked pasta
    - "Had some soup" → ERROR: Could you please be more specific, what's in the soup?

    RULES:
    - Simple single food items (apple, banana, nuts, etc.) are ALWAYS acceptable - use standard portion sizes
    - Always estimate quantities when missing but food is identifiable
    - "Pasta" = estimate 1 cup cooked pasta, "steak" = estimate 200g steak
    - Only error if completely unclear ("soup", "some food") or unrelated to eating

    For nutrition estimates use common portions. Glycemic index: vegetables/nuts (15-35), grains (25-45), white bread/rice (70-85).

    CRITICAL:
    - Set ONLY "payload" field if you can create the log (even with estimates)
    - Set ONLY "error" field if info is too vague
    - Don't just say "too vague", come up with a proper error message
    - ONLY if the query is completely unrelated to food/eating (like "I went running"), set error: "You want to log a meal but the query is not about food"
    - Single food items like "apple", "banana", "nuts", "chocolate", "espresso" ARE valid meals - use standard portions
    - Complex foods like "pasta", "steak", "bread" ARE about meals - estimate quantities if missing
    - Never set both fields

    Output valid JSON only.

    User query:
    `;
}


function createHistory(query) {
  const prompt = mealPrompt(responseSchema);
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
  const filePath = 'meal/benchmark-results/current/' + new Date().toISOString() + '.json';

  for (const sample of mealDatasetOriginal) {
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
};

main()
  .catch(console.error)
  .finally(() => {
    process.kill(process.pid);
  });


import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import ultimateSeedDataWithEmbeddings from "./meal-datasets/ultimate-seed-data-with-embeddings.json" with { type: "json" };
import mealDatasetOriginal from "./meal-datasets/meal-dataset-original.json" with { type: "json" };
import italianDishesLarge from "./meal-datasets/italian-dishes-large.json" with { type: "json" };
import { calculatePayloadMetrics, extractJSON, writeResultIncrementallyMeals } from "../utils.js";

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
    ragExamples = "\n--- SIMILAR EXAMPLES ---\n";
    top3.forEach((example, idx) => {
      ragExamples += `\nExample ${idx + 1} (similarity: ${example.similarity?.toString() || 'N/A'}):\nInput: "${example.prompt}"\nOutput: ${JSON.stringify(example.expected_output)}\n`;
    });
    ragExamples += "\n";
  }

  return `/no_think
Parse meal queries to JSON with nutrition estimates.

Schema: ${JSON.stringify(z.toJSONSchema(schema), null, 2)}

DECISION RULE:
Does query mention specific food/drink name? → PAYLOAD
No specific food/drink mentioned? → ERROR

Core Examples:
"Ate pasta" → {"payload":{"description":"Pasta (1 cup)","calories":220,"carbsGrams":43,"proteinGram":8,"fatGram":1,"glycemicIndex":60}}
"Ate bread" → {"payload":{"description":"Bread (2 slices)","calories":160,"carbsGrams":30,"proteinGram":6,"fatGram":2,"glycemicIndex":70}}
"Snacked on cheese and crackers" → {"payload":{"description":"Cheese and crackers","calories":320,"carbsGrams":22,"proteinGram":16,"fatGram":18,"glycemicIndex":45}}
"Coconut water (500ml)" → {"payload":{"description":"Coconut water 500ml","calories":90,"carbsGrams":22,"proteinGram":2,"fatGram":0,"glycemicIndex":35}}
"Restaurant meal" (no food) → {"error":"What did you order?"}
"Snack" (no food) → {"error":"What did you snack on?"}
${ragExamples}
Use Similar Examples for nutrition estimates and descriptions, but always follow the DECISION RULE above for determining payload vs error.

Food keywords: pasta, burger, pizza, steak, chicken, fish, rice, eggs, bread, cheese, yogurt, nuts, apple, banana, orange, berries, cookies, crackers, chips, popcorn, pretzels, cake, brownies, salad, sandwich, wrap, burrito, taco, soup, smoothie, juice, milk, water, coffee, tea, soda, wine, beer, protein bar, cereal, oatmeal, quinoa

GI ranges: Low (15-35), Med (40-60), High (70-85). Never return both payload and error fields.

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
  const samples = ultimateSeedDataWithEmbeddings;
  const samplesWithSimilarity = samples.map((sample) => ({
    ...sample,
    similarity: cosineSimilarity(queryEmbedding, sample.embedding),
  }));
  samplesWithSimilarity.sort((a, b) => b.similarity - a.similarity);
  return samplesWithSimilarity.slice(0, 3);
}

const main = async () => {
  await initEmbeddingModel();
  await initLlmModel();

  const filePath = 'meal/benchmark-results/rag-over-mem/' + new Date().toISOString() + '.json';

  for (const sample of [...mealDatasetOriginal, ...italianDishesLarge]) {
    const benchmarkResult = {
      prompt: sample.prompt,
      expected_output: sample.expected_output,
    };

    const queryEmbedding = await embed({ modelId: embeddingModelId, text: sample.prompt });
    let top3 = getTop3Samples(queryEmbedding);
    top3 = top3.map((sample) => {
      const { embedding, ...rest } = sample;
      return rest
    });
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
    await writeResultIncrementallyMeals(benchmarkResult, filePath);
  }

  process.kill(process.pid);
};

main().catch(console.error);

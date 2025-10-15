import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  GTE_LARGE_FP16,
  embed,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
import comprehensiveRagDatasetWithEmbeddings from "./biomarkers-datasets/comprehensive-rag-dataset-with-embeddings.json" with { type: "json" };
import biomarkerTestDataset from "./biomarkers-datasets/biomarker-test-dataset.json" with { type: "json" };
import { extractJSON, writeResultIncrementallyBiomarkers, compareBiomarkerPayloads } from "../utils.js";

export const biomarkerPayloadSchema = z.object({
  name: z.enum([
    "alanine_transaminase",
    "albumin",
    "alkaline_phosphatase",
    "aspartate_transaminase",
    "blood_oxygen_saturation",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "blood_sugar_level",
    "blood_urea_nitrogen",
    "body_fat_percentage",
    "body_mass_index",
    "body_temperature",
    "carbon_dioxide",
    "chloride",
    "complete_blood_count",
    "cortisol",
    "creatinine",
    "c_reactive_protein",
    "daily_step_count",
    "deep_sleep_duration",
    "erythrocyte_sedimentation_rate",
    "estimated_glomerular_filtration_rate",
    "free_thyroxine",
    "gamma_glutamyl_transferase",
    "glial_fibrillary_acidic_protein",
    "heart_rate",
    "heart_rate_variability",
    "hematocrit",
    "hemoglobin",
    "hemoglobin_a1c",
    "high_density_lipoprotein_cholesterol",
    "light_sleep_duration",
    "low_density_lipoprotein_cholesterol",
    "mean_corpuscular_hemoglobin",
    "mean_corpuscular_hemoglobin_concentration",
    "mean_corpuscular_volume",
    "mean_platelet_volume",
    "platelet_count",
    "potassium",
    "red_blood_cell_count",
    "red_blood_cell_distribution_width",
    "rem_sleep_duration",
    "resting_heart_rate",
    "sleep_duration",
    "sleep_quality_score",
    "sodium",
    "testosterone",
    "thyroid_stimulating_hormone",
    "total_bilirubin",
    "cholesterol",
    "total_cholesterol",
    "triglyceride",
    "troponin",
    "ubiquitin_carboxy_terminal_hydrolase_l1",
    "uric_acid",
    "vitamin_d_25_hydroxy",
    "waist_circumference",
    "waist_to_hip_ratio",
    "white_blood_cell_count",
    "muscle_mass",
    "blood_oxygen",
    "oxygen_saturation",
    "spo2",
    "lean_body_mass",
    "skeletal_muscle_mass",
    "total_cholesterol_all",
    "body_weight",
    "vo2_max",
  ]),
  value: z.number(),
  unit: z.string(),
});

const responseSchema = z.object({
  error: z.string().optional(),
  payload: z.array(biomarkerPayloadSchema).optional(),
});

function biomarkerPrompt(schema, top3) {
  // Build RAG examples section
  let ragExamples = "";
  if (top3 && top3.length > 0) {
    ragExamples = "\n--- SIMILAR EXAMPLES (similarity) ---\n";
    top3.forEach((example, idx) => {
      const sim = example.similarity ? (example.similarity * 100).toFixed(4) : 'N/A';
      ragExamples += `${idx + 1}. [${sim}%] "${example.prompt}" → ${JSON.stringify(example.expected_output)}\n`;
    });
  }

  return `/no_think
Parse health measurement queries into JSON matching this schema:

${JSON.stringify(z.toJSONSchema(schema))}

--- BASELINE EXAMPLES ---
"Heart rate 72 bpm" → {"payload":[{"name":"heart_rate","value":72,"unit":"bpm"}]}
"BP 120/80 mmHg" → {"payload":[{"name":"blood_pressure_systolic","value":120,"unit":"mmHg"},{"name":"blood_pressure_diastolic","value":80,"unit":"mmHg"}]}
"Cholesterol is high" → {"payload":[{"name":"total_cholesterol","value":240,"unit":"mg/dL"}]}
"My blood sugar is low" → {"payload":[{"name":"blood_sugar_level","value":70,"unit":"mg/dL"}]}
"Blood pressure is normal" → {"payload":[{"name":"blood_pressure_systolic","value":120,"unit":"mmHg"},{"name":"blood_pressure_diastolic","value":80,"unit":"mmHg"}]}
"My heart rate is high" → {"payload":[{"name":"heart_rate","value":95,"unit":"bpm"}]}
"Checked my glucose" → {"error":"What was your glucose reading?"}
"Went for a run" → {"error":"You want to log a biomarker but the query is not about health measurements"}
${ragExamples}
RULES:
1. CRITICAL: Queries with qualitative descriptions (high/low/normal/elevated/decreased) ARE health measurements - estimate appropriate values
2. Return PAYLOAD if query has explicit values OR qualitative health descriptions
3. Return ERROR only if query mentions measurement name WITHOUT any value or description (e.g., "checked my glucose")
4. Return ERROR if query is completely unrelated to health (e.g., "went for a run")
5. ESTIMATION GUIDE:
   - "high" → estimate high-range values (e.g., cholesterol=240, heart rate=95, blood sugar=180)
   - "low" → estimate low-range values (e.g., cholesterol=155, heart rate=55, blood sugar=70)
   - "normal" → estimate mid-range normal values (e.g., cholesterol=185, heart rate=72, blood sugar=95)
6. Biomarker "name" MUST match schema enum exactly
7. Blood pressure: Always split into systolic + diastolic when both present
8. Output valid JSON only - never schema definitions
9. When units are provided in query, "unit" MUST be EXACTLY as provided in the query
10. When estimating, use standard medical units (mg/dL for cholesterol/glucose, bpm for heart rate, mmHg for blood pressure)

User query:`;
}

function createHistory(query, top3) {
  const prompt = biomarkerPrompt(responseSchema, top3);
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
  const samplesWithSimilarity = comprehensiveRagDatasetWithEmbeddings.map((sample) => ({
    ...sample,
    similarity: cosineSimilarity(queryEmbedding, sample.embedding),
  }));
  samplesWithSimilarity.sort((a, b) => b.similarity - a.similarity);
  return samplesWithSimilarity.slice(0, 3);
}

const main = async () => {
  await initEmbeddingModel();
  await initLlmModel();

  const filePath = 'biomarkers/benchmark-results/rag-over-mem/' + new Date().toISOString() + '.json';

  for (const sample of biomarkerTestDataset) {
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
        // Compare biomarker arrays for quality measurement
        const comparison = compareBiomarkerPayloads(
          sample.expected_output.payload,
          parsedResult.payload
        );
        benchmarkResult.biomarkerComparison = comparison;
        
        // Define threshold: 80% or higher = truthy, >0% but <80% = partial
        const MATCH_THRESHOLD = 80;
        
        if (comparison.matchPercentage >= MATCH_THRESHOLD) {
          benchmarkResult.classification = "truthy_payload";
        } else if (comparison.matchPercentage > 0) {
          // Some biomarkers match but not enough
          benchmarkResult.classification = "partial_payload";
        } else {
          // No biomarkers match at all
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
    writeResultIncrementallyBiomarkers(benchmarkResult, filePath);
  }

  process.kill(process.pid);
};

main().catch(console.error);


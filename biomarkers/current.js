import {
  loadModel,
  QWEN_3_1_7B_INST_Q4,
  completion,
} from "@tetherto/qvac-sdk";
import { z } from "zod";
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

function biomarkerPrompt(schema) {
  return `/no_think
    You are given a schema for a biomarker tool call and you need to fill it based on the user query. Here's the schema:

    ${JSON.stringify(z.toJSONSchema(schema))}

    GOOD examples (specific with values and units):
    - "My heart rate is 72 bpm" ✓
    - "Blood pressure 120/80 mmHg" ✓ (systolic 120, diastolic 80)
    - "Weight is 70.5 kg this morning" ✓
    - "Glucose level 95 mg/dL after fasting" ✓
    - "My HRV was 30" ✓ (Heart Rate Variability)

    BAD examples (missing info, estimate anyway):
    - "My total cholesterol is high" → Estimate: 200 mg/dL (high)
    - "My total cholesterol is low" → Estimate: 125 mg/dL (low)
    - "My heart rate is high" → Estimate: 90 bpm (high)
    - "Blood pressure is normal" → Estimate: systolic 120 mmHg, diastolic 80 mmHg
    - "I checked my glucose" → ERROR: What was the reading?

    RULES:
    - Need biomarker name, value, and unit
    - Can estimate typical values for vague descriptions (high/low/normal)
    - If no value given at all, use error field
    - Common units: bpm, mmHg, kg, mg/dL, °F/°C

    Categories: cardiovascular (heart rate, BP), metabolic (glucose, ketones), physical (weight, temp).

    CRITICAL:
    - Set ONLY "payload" field if you can create the log (even with estimates)
    - Set ONLY "error" field if info is too vague to identify
    - Don't just say "too vague", come up with a proper error message
    - ONLY if the query is completely unrelated to health/measurements (like "I went running"), set error: "You want to log a biomarker but the query is not about health measurements"
    - Never set both fields
    - ENSURE THE NAME FIELD ADHERES TO THE SCHEMA

    Output valid JSON only.

    User query:
    `;
}


function createHistory(query) {
  const prompt = biomarkerPrompt(responseSchema);
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
  const filePath = 'biomarkers/benchmark-results/current/' + new Date().toISOString() + '.json';

  for (const sample of biomarkerTestDataset) {
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
};

main()
  .catch(console.error)
  .finally(() => {
    process.kill(process.pid);
  });


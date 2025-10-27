import workoutRagDataset from "./workout-rag-dataset.json" with { type: "json" };
import fs from "fs";

// Terms that should consistently return errors
const VAGUE_TERMS = [
  "gym", "workout", "exercise", "exercised", "cardio", "stairs", 
  "sports", "worked out", "training", "fitness"
];

// Terms that should consistently return payloads
const SPECIFIC_ACTIVITIES = [
  "run", "ran", "jog", "swim", "swam", "yoga", "pilates", "walk",
  "cycle", "bike", "row", "bench press", "squat", "deadlift", "pushup",
  "pullup", "plank", "lunge", "basketball", "soccer", "tennis", "golf"
];

console.log("🔍 Analyzing Workout RAG Dataset Quality\n");
console.log(`Total entries: ${workoutRagDataset.length}\n`);

// Track contradictions
const contradictions = [];
const vagueCases = [];
const specificCases = [];
const workoutTypeVariants = new Map();
const extremeCalories = [];

for (const entry of workoutRagDataset) {
  const prompt = entry.prompt.toLowerCase();
  const hasPayload = !!entry.expected_output.payload;
  const hasError = !!entry.expected_output.error;
  
  // Check for vague term handling
  for (const term of VAGUE_TERMS) {
    if (prompt === term || prompt === `${term}s` || prompt.match(new RegExp(`^${term}[\\s,.]`))) {
      if (hasPayload) {
        vagueCases.push({
          id: entry.id,
          prompt: entry.prompt,
          issue: `Vague term "${term}" returns payload instead of error`,
          output: entry.expected_output
        });
      }
    }
  }
  
  // Check for specific activity handling
  for (const activity of SPECIFIC_ACTIVITIES) {
    if (prompt.includes(activity)) {
      if (hasError) {
        specificCases.push({
          id: entry.id,
          prompt: entry.prompt,
          issue: `Specific activity "${activity}" returns error instead of payload`,
          output: entry.expected_output
        });
      }
    }
  }
  
  // Track workout type variants
  if (hasPayload && entry.expected_output.payload.workoutType) {
    const type = entry.expected_output.payload.workoutType;
    if (!workoutTypeVariants.has(type)) {
      workoutTypeVariants.set(type, []);
    }
    workoutTypeVariants.get(type).push(entry.prompt);
  }
  
  // Check for extreme/unrealistic calorie values
  if (hasPayload) {
    const payload = entry.expected_output.payload;
    if (payload.caloriesBurned && payload.durationMinutes) {
      const rate = payload.caloriesBurned / payload.durationMinutes;
      
      // Flag if rate is too high (>20 cal/min) or too low (<2 cal/min)
      if (rate > 20 || rate < 2) {
        extremeCalories.push({
          id: entry.id,
          prompt: entry.prompt,
          duration: payload.durationMinutes,
          calories: payload.caloriesBurned,
          rate: rate.toFixed(2),
          issue: rate > 20 ? "Unrealistically high calorie rate" : "Unrealistically low calorie rate"
        });
      }
    }
  }
}

// Find contradictory examples (same prompt, different outputs)
const promptMap = new Map();
for (const entry of workoutRagDataset) {
  const normalizedPrompt = entry.prompt.toLowerCase().trim();
  if (!promptMap.has(normalizedPrompt)) {
    promptMap.set(normalizedPrompt, []);
  }
  promptMap.get(normalizedPrompt).push(entry);
}

for (const [prompt, entries] of promptMap) {
  if (entries.length > 1) {
    // Check if outputs differ
    const outputs = entries.map(e => JSON.stringify(e.expected_output));
    const uniqueOutputs = [...new Set(outputs)];
    if (uniqueOutputs.length > 1) {
      contradictions.push({
        prompt: entries[0].prompt,
        count: entries.length,
        entries: entries.map(e => ({
          id: e.id,
          output: e.expected_output
        }))
      });
    }
  }
}

// Report findings
console.log("=" .repeat(80));
console.log("📊 ANALYSIS RESULTS");
console.log("=".repeat(80));

console.log(`\n❌ CONTRADICTIONS (same prompt, different outputs): ${contradictions.length}`);
if (contradictions.length > 0) {
  console.log("\nTop 10 contradictions:");
  contradictions.slice(0, 10).forEach(c => {
    console.log(`\n  "${c.prompt}" (${c.count} entries):`);
    c.entries.forEach((e, idx) => {
      console.log(`    ${idx + 1}. [${e.id}] ${JSON.stringify(e.output)}`);
    });
  });
}

console.log(`\n⚠️  VAGUE TERMS RETURNING PAYLOAD: ${vagueCases.length}`);
if (vagueCases.length > 0) {
  console.log("\nExamples (first 15):");
  vagueCases.slice(0, 15).forEach(c => {
    console.log(`  "${c.prompt}" → ${c.issue}`);
  });
}

console.log(`\n⚠️  SPECIFIC ACTIVITIES RETURNING ERROR: ${specificCases.length}`);
if (specificCases.length > 0) {
  console.log("\nExamples (first 10):");
  specificCases.slice(0, 10).forEach(c => {
    console.log(`  "${c.prompt}" → ${c.issue}`);
  });
}

console.log(`\n🔥 EXTREME CALORIE RATES: ${extremeCalories.length}`);
if (extremeCalories.length > 0) {
  console.log("\nExamples (first 10):");
  extremeCalories.slice(0, 10).forEach(c => {
    console.log(`  "${c.prompt}"`);
    console.log(`    ${c.duration} min, ${c.calories} cal = ${c.rate} cal/min (${c.issue})`);
  });
}

console.log(`\n📋 WORKOUT TYPE VARIANTS: ${workoutTypeVariants.size}`);
const sortedTypes = [...workoutTypeVariants.entries()].sort((a, b) => b[1].length - a[1].length);
console.log("\nTop 20 workout types by frequency:");
sortedTypes.slice(0, 20).forEach(([type, prompts]) => {
  console.log(`  ${type}: ${prompts.length} entries`);
});

// Find non-standard workout types
const standardTypes = [
  "running", "walking", "cycling", "swimming", "rowing", "elliptical",
  "strength training", "weightlifting", "calisthenics",
  "yoga", "pilates", "stretching",
  "HIIT", "circuit training", "CrossFit",
  "mixed training"
];

console.log("\n🔍 NON-STANDARD WORKOUT TYPES (should be normalized):");
const nonStandardTypes = sortedTypes.filter(([type]) => {
  return !standardTypes.includes(type) && 
         !type.match(/^(basketball|soccer|tennis|golf|football|baseball|volleyball|hockey)/);
});

nonStandardTypes.slice(0, 30).forEach(([type, prompts]) => {
  console.log(`  "${type}": ${prompts.length} entries`);
});

// Create filtered dataset
console.log("\n" + "=".repeat(80));
console.log("🔧 CREATING FILTERED DATASET");
console.log("=".repeat(80));

const filteredDataset = workoutRagDataset.filter(entry => {
  // Remove duplicates (keep first occurrence)
  const normalizedPrompt = entry.prompt.toLowerCase().trim();
  const isFirst = promptMap.get(normalizedPrompt)[0].id === entry.id;
  if (!isFirst) return false;
  
  // Remove vague terms that return payloads
  const isVagueWithPayload = vagueCases.some(v => v.id === entry.id);
  if (isVagueWithPayload) return false;
  
  // Remove extreme calorie rates (likely data errors)
  const hasExtremeCalories = extremeCalories.some(e => e.id === entry.id);
  if (hasExtremeCalories) return false;
  
  return true;
});

console.log(`\n✅ Original dataset: ${workoutRagDataset.length} entries`);
console.log(`✅ Filtered dataset: ${filteredDataset.length} entries`);
console.log(`❌ Removed: ${workoutRagDataset.length - filteredDataset.length} entries`);

// Save filtered dataset
fs.writeFileSync(
  "./workouts/workout-rag-dataset-filtered.json",
  JSON.stringify(filteredDataset, null, 2)
);

console.log("\n💾 Filtered dataset saved to: workout-rag-dataset-filtered.json");

// Save detailed report
const report = {
  summary: {
    total: workoutRagDataset.length,
    filtered: filteredDataset.length,
    removed: workoutRagDataset.length - filteredDataset.length,
    contradictions: contradictions.length,
    vagueCases: vagueCases.length,
    specificCases: specificCases.length,
    extremeCalories: extremeCalories.length
  },
  contradictions,
  vagueCases,
  specificCases,
  extremeCalories,
  workoutTypeVariants: Object.fromEntries(
    sortedTypes.map(([type, prompts]) => [type, prompts.length])
  ),
  nonStandardTypes: nonStandardTypes.map(([type, prompts]) => ({
    type,
    count: prompts.length,
    examples: prompts.slice(0, 3)
  }))
};

fs.writeFileSync(
  "./workouts/dataset-quality-report.json",
  JSON.stringify(report, null, 2)
);

console.log("📄 Detailed report saved to: dataset-quality-report.json");
console.log("\n✨ Analysis complete!\n");


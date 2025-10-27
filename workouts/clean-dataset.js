import workoutRagDataset from "./workout-rag-dataset.json" with { type: "json" };
import fs from "fs";

console.log("🧹 Cleaning Workout RAG Dataset\n");
console.log(`Original entries: ${workoutRagDataset.length}\n`);

// Define TRULY vague prompts (standalone terms only)
const TRULY_VAGUE = [
  "gym", "workout", "exercise", "exercised", "worked out", 
  "cardio", "training", "fitness", "stairs", "sports"
];

// Normalize workout types to standard terms
const WORKOUT_TYPE_MAPPING = {
  // Normalize variants to standard types
  "gym session": "strength training",
  "leg day": "strength training",
  "arm day": "strength training", 
  "upper body": "strength training",
  "lower body": "strength training",
  "full body strength": "strength training",
  "weight training": "weightlifting",
  "weights": "weightlifting",
  "bodyweight training": "calisthenics",
  "bodyweight": "calisthenics",
  
  // Cardio variants
  "cardio": "cardio",
  "jogging": "running",
  "jog": "running",
  "run": "running",
  "treadmill": "running",
  "trail running": "running",
  "sprints": "running",
  
  "biking": "cycling",
  "bike": "cycling",
  "road cycling": "cycling",
  "mountain biking": "cycling",
  "indoor cycling": "cycling",
  "spin": "cycling",
  "gravel cycling": "cycling",
  "bike touring": "cycling",
  
  "elliptical machine": "elliptical",
  "eliptical": "elliptical",
  
  "swimming laps": "swimming",
  "lap swimming": "swimming",
  
  // HIIT variants
  "hiit": "HIIT",
  "circuit": "circuit training",
  "circuits": "circuit training",
  "crossfit": "CrossFit",
  
  // Flexibility
  "stretches": "stretching",
  "stretch": "stretching",
  
  // Combat sports
  "boxing": "boxing",
  "kickboxing": "boxing",
  "muay thai": "boxing",
  "martial arts": "martial arts",
  
  // Low intensity (should be kept as valid workouts)
  "meditation": "meditation",
  "tai chi": "tai chi",
  "qigong": "qigong",
  "breathwork": "stretching",
  
  // Sports - keep specific
  "badminton": "badminton",
  "table tennis": "table tennis", 
  "racquetball": "racquetball",
  "squash": "squash",
  "handball": "handball",
  "pickleball": "pickleball",
  "tennis drills": "tennis",
  "squash drills": "squash",
  
  // Other
  "stair climbing": "stair climbing",
  "stairs": "stair climbing",
  "barre": "barre",
  "mobility": "stretching",
  "recovery": "stretching"
};

let cleaned = 0;
let removed = 0;
const issues = {
  trulyVague: [],
  invalidData: [],
  extremeCalories: [],
  negativeDuration: []
};

const cleanedDataset = workoutRagDataset.filter(entry => {
  const prompt = entry.prompt.toLowerCase().trim();
  const hasPayload = !!entry.expected_output.payload;
  const hasError = !!entry.expected_output.error;
  
  // Remove truly vague standalone terms that have payloads
  if (hasPayload && TRULY_VAGUE.some(term => 
    prompt === term || 
    prompt === `${term}s` ||
    prompt === `did ${term}` ||
    prompt === `i ${term}`
  )) {
    issues.trulyVague.push(entry.prompt);
    removed++;
    return false;
  }
  
  // Remove invalid/edge case data
  if (hasPayload) {
    const p = entry.expected_output.payload;
    
    // Remove negative or zero durations
    if (p.durationMinutes !== undefined && p.durationMinutes <= 0) {
      issues.negativeDuration.push(entry.prompt);
      removed++;
      return false;
    }
    
    // Remove extreme calorie rates (but allow very low rates for meditation/stretching)
    if (p.caloriesBurned && p.durationMinutes) {
      const rate = p.caloriesBurned / p.durationMinutes;
      const workoutType = p.workoutType?.toLowerCase() || "";
      
      // Allow low rates for meditation, yoga nidra, restorative activities
      const isLowIntensity = workoutType.includes("meditation") || 
                             workoutType.includes("nidra") ||
                             workoutType.includes("restorative") ||
                             workoutType.includes("massage") ||
                             workoutType.includes("standing desk");
      
      // Remove if rate is unrealistically high (>20 cal/min)
      if (rate > 20) {
        issues.extremeCalories.push(`${entry.prompt} (${rate.toFixed(1)} cal/min)`);
        removed++;
        return false;
      }
      
      // Remove very low intensity activities (<1 cal/min) unless they're meditation-like
      if (rate < 1 && !isLowIntensity) {
        issues.extremeCalories.push(`${entry.prompt} (${rate.toFixed(1)} cal/min)`);
        removed++;
        return false;
      }
    }
  }
  
  // Remove "planning to" or "will" statements (future, not logged)
  if ((prompt.includes("planning") || prompt.includes("will go") || prompt.includes("tomorrow")) && hasError) {
    issues.invalidData.push(entry.prompt);
    removed++;
    return false;
  }
  
  cleaned++;
  return true;
});

// Now normalize workout types in the cleaned dataset
const normalizedDataset = cleanedDataset.map(entry => {
  if (entry.expected_output.payload && entry.expected_output.payload.workoutType) {
    const originalType = entry.expected_output.payload.workoutType;
    const normalizedType = WORKOUT_TYPE_MAPPING[originalType.toLowerCase()] || originalType;
    
    return {
      ...entry,
      expected_output: {
        ...entry.expected_output,
        payload: {
          ...entry.expected_output.payload,
          workoutType: normalizedType
        }
      }
    };
  }
  return entry;
});

console.log("📊 CLEANING RESULTS");
console.log("=".repeat(60));
console.log(`✅ Kept: ${cleaned} entries`);
console.log(`❌ Removed: ${removed} entries`);
console.log(`\nRemoval reasons:`);
console.log(`  • Truly vague terms: ${issues.trulyVague.length}`);
console.log(`  • Invalid data: ${issues.invalidData.length}`);
console.log(`  • Extreme calories: ${issues.extremeCalories.length}`);
console.log(`  • Negative duration: ${issues.negativeDuration.length}`);

if (issues.trulyVague.length > 0) {
  console.log(`\n🗑️  Removed vague terms:`);
  issues.trulyVague.forEach(p => console.log(`    - "${p}"`));
}

if (issues.negativeDuration.length > 0) {
  console.log(`\n🗑️  Removed negative/zero duration:`);
  issues.negativeDuration.forEach(p => console.log(`    - "${p}"`));
}

if (issues.invalidData.length > 0) {
  console.log(`\n🗑️  Removed invalid data:`);
  issues.invalidData.forEach(p => console.log(`    - "${p}"`));
}

// Save cleaned dataset
fs.writeFileSync(
  "./workouts/workout-rag-dataset-cleaned.json",
  JSON.stringify(normalizedDataset, null, 2)
);

console.log(`\n💾 Cleaned dataset saved to: workout-rag-dataset-cleaned.json`);
console.log(`📈 Final count: ${normalizedDataset.length} entries`);

// Create a summary of changes
const summary = {
  original: workoutRagDataset.length,
  cleaned: normalizedDataset.length,
  removed: removed,
  issues: {
    trulyVague: issues.trulyVague,
    invalidData: issues.invalidData,
    extremeCalories: issues.extremeCalories,
    negativeDuration: issues.negativeDuration
  },
  workoutTypeMappings: Object.keys(WORKOUT_TYPE_MAPPING).length
};

fs.writeFileSync(
  "./workouts/cleaning-summary.json",
  JSON.stringify(summary, null, 2)
);

console.log(`📄 Summary saved to: cleaning-summary.json`);
console.log("\n✨ Cleaning complete!\n");




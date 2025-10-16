# Workout Logging Benchmark

## Overview

This directory contains the workout logging benchmark implementation that tests the model's ability to parse natural language workout descriptions and extract structured data.

## Files

- **`current.js`** - Main benchmark runner that processes the workout dataset
- **`workout-test-dataset.json`** - Comprehensive test dataset with 50 workout scenarios

## Test Dataset Structure

Each entry in the dataset follows this structure:

```json
{
  "id": "entry_XXX_hash",
  "prompt": "User's workout description",
  "expected_output": {
    "payload": { /* Expected structured data */ }
    // OR
    "error": "Expected error message"
  }
}
```

## Workout Payload Schema

The model extracts the following fields:

- **`workoutType`** (string, optional): Type of workout (running, cycling, strength training, etc.)
- **`description`** (string, optional): Brief description of the workout
- **`durationMinutes`** (number, optional): Duration in minutes
- **`caloriesBurned`** (number, optional): Estimated calories burned
- **`intensityLevel`** (string, optional): Intensity level (low, moderate, high)
- **`exercises`** (array, optional): Array of exercise objects with:
  - `name` (string): Exercise name
  - `sets` (number, optional): Number of sets
  - `reps` (number, optional): Number of reps
  - `weight` (number, optional): Weight amount
  - `weightUnit` (string, optional): Weight unit (kg, lbs)
  - `durationMinutes` (number, optional): Exercise duration

## Metrics Calculation

The workout metrics are more sophisticated than meal metrics due to the complexity of the schema:

### 1. Numeric Field Errors (Normalized)
- **Duration Error**: Normalized error for `durationMinutes`
- **Calories Error**: Normalized error for `caloriesBurned`

### 2. String Field Matches (0-1 score)
- **Workout Type Match**: Fuzzy matching for workout type
- **Description Match**: Fuzzy matching for description
- **Intensity Level Match**: Fuzzy matching for intensity

### 3. Exercise Array Score (0-1)
- Matches exercises by name
- Scores each exercise based on:
  - Name similarity (weighted 2x)
  - Sets/reps/weight accuracy
  - Weight unit match

### 4. Overall Score
Weighted combination:
- **50%** Numeric accuracy (duration + calories)
- **20%** String matches (workout type, description, intensity)
- **30%** Exercise accuracy

## String Matching Algorithm

The string matching uses a three-tier approach:

1. **Exact match** → 1.0 score
2. **Partial match** (one contains other) → 0.7 score
3. **Word overlap** (Jaccard similarity) → 0.0-1.0 score

## Benchmark Results Summary

The benchmark produces a comprehensive summary including:

- **Classification counts**: truthy_payload, falsy_payload, truthy_error, falsy_error
- **Accuracy metrics**: Overall accuracy, payload accuracy, error accuracy
- **Average normalized error**: For numeric fields
- **Average overall score**: Weighted combination of all metrics
- **Average string match scores**: For workout type, description, intensity
- **Average exercises score**: For exercise array matching

## Running the Benchmark

```bash
node workouts/current.js
```

Results are saved to:
```
workouts/benchmark-results/current/[timestamp].json
```

## Test Coverage

The 50 test cases cover:

1. **Detailed workouts** (5): Full information provided
2. **Vague prompts** (3): Tests estimation capabilities
3. **Error cases** (7): Too vague or unrelated queries
4. **Complex workouts** (1): Multi-exercise strength training
5. **Intensity variations** (2): Low and high intensity
6. **Mixed activities** (2): Combined cardio and strength
7. **Unit variations** (1): Pounds vs kilograms
8. **Minimal info** (3): Very brief descriptions
9. **Common types** (7): Generic workout categories
10. **Various cardio** (4): Hiking, rowing, spin, jump rope
11. **Class-based** (5): Pilates, CrossFit, boxing, martial arts
12. **Sports** (4): Tennis, soccer, basketball, climbing
13. **Edge cases** (2): Ambiguous single-word prompts
14. **Bodyweight** (1): Calisthenics without weights
15. **Casual language** (1): Natural/slang descriptions

## Key Features

1. **Estimation**: Model should estimate missing duration and calories based on workout type
2. **Fuzzy Matching**: Accounts for variations in terminology (e.g., "running" vs "run")
3. **Exercise Matching**: Intelligently matches exercises by name across arrays
4. **Comprehensive Scoring**: Multiple metrics provide detailed quality assessment
5. **Incremental Results**: Results are written incrementally with running statistics


# Workout RAG-over-mem Setup

This document explains the RAG-over-mem implementation for workouts.

## Overview

The RAG-over-mem (Retrieval-Augmented Generation over Memory) approach enhances the workout parsing model by providing similar examples from a training dataset at inference time. This helps the model better understand workout patterns and improve accuracy.

## Files Created

### 1. `generate-embeddings.js`
Generates embeddings for the workout RAG dataset using the GTE-Large embedding model.

**Input:** `workout-rag-dataset.json` (large training dataset)
**Output:** `workouts-datasets/workout-rag-dataset-with-embeddings.json`

**Usage:**
```bash
node workouts/generate-embeddings.js
```

This will:
- Load the GTE-Large FP16 embedding model
- Process all entries in the workout RAG dataset
- Generate embeddings for each prompt
- Save the dataset with embeddings

**Note:** This only needs to be run once (or when the RAG dataset is updated).

### 2. `rag-over-mem.js`
Main RAG-powered benchmark runner that uses similarity search to retrieve relevant examples.

**Features:**
- Loads both embedding and LLM models
- For each test query:
  1. Generates an embedding for the query
  2. Finds top 3 most similar examples from the RAG dataset using cosine similarity
  3. Includes these examples in the prompt
  4. Runs inference with the LLM
  5. Calculates metrics and saves results

**Usage:**
```bash
node workouts/rag-over-mem.js
```

Results are saved to: `workouts/benchmark-results/rag-over-mem/[timestamp].json`

## How It Works

### Similarity Search
The system uses cosine similarity to find the most relevant training examples:

1. **Query Embedding:** Convert the user's workout query into a vector embedding
2. **Compare:** Calculate cosine similarity between the query and all training examples
3. **Retrieve:** Select the top 3 most similar examples
4. **Augment:** Include these examples in the prompt to guide the model

### Prompt Structure
The RAG-enhanced prompt includes:
- Schema definition
- Decision rules (when to return payload vs error)
- Core hardcoded examples
- **RAG Examples:** Top 3 similar examples with similarity scores
- Workout-specific guidelines (calorie estimation, intensity levels, etc.)

Example:
```
--- SIMILAR EXAMPLES ---
1. [95.23%] "Ran 5k in 28 minutes" → {"payload":{...}}
2. [92.15%] "Morning jog for 20 minutes" → {"payload":{...}}
3. [88.44%] "Fast 10k run in 47 minutes" → {"payload":{...}}
```

## Comparison with Other Approaches

### vs. Current (No RAG)
- **Current:** Uses only hardcoded examples in the prompt
- **RAG-over-mem:** Dynamically retrieves relevant examples based on query similarity
- **Expected benefit:** Better handling of edge cases and workout variations

### vs. Other Categories
This implementation follows the same pattern as:
- `medications/rag-over-mem.js` - Medication logging with RAG
- `biomarkers/rag-over-mem.js` - Biomarker logging with RAG
- `symptoms/rag-over-mem.js` - Symptom logging with RAG
- `meal/rag-over-mem.js` - Meal logging with RAG

## Workout-Specific Features

### Calorie Estimation
The system estimates calories based on workout type and duration:
- Running: ~10 cal/min
- Walking: ~5 cal/min
- Cycling: ~8 cal/min
- Swimming: ~11 cal/min
- Strength training: ~6 cal/min
- Yoga: ~3 cal/min
- HIIT: ~12 cal/min

### Intensity Detection
Keywords are used to determine intensity:
- **High:** intense, hard, pushed, fast, vigorous, heavy, max
- **Moderate:** normal, regular, steady, comfortable (default)
- **Low:** easy, light, gentle, recovery, slow

### Exercise Array
For strength training workouts, the system extracts structured exercise data:
- Exercise name
- Sets and reps
- Weight and weight unit (kg/lbs)
- Duration (if applicable)

## Metrics

The system calculates comprehensive metrics for workout payloads:

1. **Numeric Errors:**
   - Duration error (normalized)
   - Calories error (normalized)

2. **String Matches:**
   - Workout type match (0-1)
   - Description match (0-1)
   - Intensity level match (0-1)

3. **Exercise Array Score:**
   - Matches exercises by name
   - Scores each exercise on sets, reps, weight, etc.

4. **Overall Score:**
   - 50% numeric accuracy
   - 20% string matches
   - 30% exercises score

## Dataset Information

### RAG Training Dataset
- **File:** `workout-rag-dataset.json`
- **Size:** ~14,125 lines (large training set)
- **Purpose:** Provides examples for similarity search

### Test Dataset
- **File:** `workouts-datasets/workout-test-dataset.json`
- **Size:** 50+ test cases
- **Purpose:** Benchmark evaluation

## Running the Complete Pipeline

1. **First time setup:** Generate embeddings
   ```bash
   node workouts/generate-embeddings.js
   ```

2. **Run RAG benchmark:**
   ```bash
   node workouts/rag-over-mem.js
   ```

3. **Compare results:**
   - Current approach: `workouts/benchmark-results/current/`
   - RAG approach: `workouts/benchmark-results/rag-over-mem/`

## Expected Improvements

RAG should improve performance on:
- Varied workout descriptions (casual language, abbreviations)
- Uncommon exercises or workout types
- Complex strength training sessions with multiple exercises
- Edge cases not covered by hardcoded examples

## Technical Details

- **Embedding Model:** GTE-Large FP16 (1024 dimensions)
- **LLM Model:** Qwen 3.1 7B Instruct Q4
- **Similarity Metric:** Cosine similarity
- **Top-K:** 3 most similar examples
- **Context Size:** 2048 tokens
- **GPU Layers:** 999 (full GPU acceleration)


export function extractJSON(text) {
  const firstBracket = text.indexOf("{");
  const lastBracket = text.lastIndexOf("}");

  if (
    firstBracket === -1 ||
    lastBracket === -1 ||
    firstBracket >= lastBracket
  ) {
    throw new Error("No valid JSON found in response");
  }

  return text.substring(firstBracket, lastBracket + 1);
}

export function calculatePayloadMetrics(expected, actual) {
  const metrics = {
    caloriesError: calculateNormalizedError(expected.calories, actual.calories),
    carbsError: calculateNormalizedError(
      expected.carbsGrams,
      actual.carbsGrams
    ),
    proteinError: calculateNormalizedError(
      expected.proteinGram,
      actual.proteinGram
    ),
    fatError: calculateNormalizedError(expected.fatGram, actual.fatGram),
    glycemicIndexError: calculateNormalizedError(
      expected.glycemicIndex,
      actual.glycemicIndex
    ),
    averageNormalizedError: 0,
  };

  metrics.averageNormalizedError =
    (metrics.caloriesError +
      metrics.carbsError +
      metrics.proteinError +
      metrics.fatError +
      metrics.glycemicIndexError) /
    5;

  return metrics;
}

export function calculateNormalizedError(expected, actual) {
  if (expected === 0) {
    return actual === 0 ? 0 : 1;
  }
  return Math.abs((expected - actual) / expected);
}

/**
 * Calculate metrics for workout payloads
 * @param {Object} expected - Expected workout payload
 * @param {Object} actual - Actual workout payload from model
 * @returns {Object} Metrics including numeric errors, string matches, and exercise scores
 */
export function calculateWorkoutPayloadMetrics(expected, actual) {
  const metrics = {
    // Numeric field errors (normalized)
    durationError: calculateNormalizedError(
      expected.durationMinutes, 
      actual.durationMinutes
    ),
    caloriesError: calculateNormalizedError(
      expected.caloriesBurned, 
      actual.caloriesBurned
    ),
    
    // String field matches (exact or fuzzy)
    workoutTypeMatch: calculateStringMatch(
      expected.workoutType, 
      actual.workoutType
    ),
    descriptionMatch: calculateStringMatch(
      expected.description, 
      actual.description
    ),
    intensityLevelMatch: calculateStringMatch(
      expected.intensityLevel, 
      actual.intensityLevel
    ),
    
    // Exercise array comparison
    exercisesScore: calculateExercisesScore(
      expected.exercises, 
      actual.exercises
    ),
    
    // Overall metrics
    averageNormalizedError: 0,
    overallScore: 0
  };

  // Calculate average numeric error (for duration and calories)
  const numericErrors = [
    metrics.durationError, 
    metrics.caloriesError
  ].filter(e => e !== null && !isNaN(e));
  
  metrics.averageNormalizedError = numericErrors.length > 0
    ? numericErrors.reduce((sum, e) => sum + e, 0) / numericErrors.length
    : 0;

  // Calculate overall score (combining numeric accuracy and string matches)
  const stringMatches = [
    metrics.workoutTypeMatch,
    metrics.descriptionMatch,
    metrics.intensityLevelMatch
  ].filter(m => m !== null);
  
  const avgStringMatch = stringMatches.length > 0
    ? stringMatches.reduce((sum, m) => sum + m, 0) / stringMatches.length
    : 1;
  
  // Overall score: 50% numeric accuracy, 20% string matches, 30% exercises
  const numericAccuracy = 1 - metrics.averageNormalizedError;
  metrics.overallScore = (
    (numericAccuracy * 0.5) + 
    (avgStringMatch * 0.2) + 
    (metrics.exercisesScore * 0.3)
  );

  return metrics;
}

/**
 * Calculate string similarity between expected and actual strings
 * @param {string} expected - Expected string value
 * @param {string} actual - Actual string value
 * @returns {number|null} Similarity score (0-1) or null if not applicable
 */
export function calculateStringMatch(expected, actual) {
  // Handle missing values
  if (expected === undefined && actual === undefined) return null; // Not applicable
  if (expected === undefined || actual === undefined) return 0; // Mismatch
  
  // Normalize strings for comparison
  const expStr = String(expected).toLowerCase().trim();
  const actStr = String(actual).toLowerCase().trim();
  
  // Exact match
  if (expStr === actStr) return 1;
  
  // Check if one contains the other (partial match)
  if (expStr.includes(actStr) || actStr.includes(expStr)) {
    return 0.7; // Partial match gets 70%
  }
  
  // Simple similarity based on shared words
  const expWords = new Set(expStr.split(/\s+/));
  const actWords = new Set(actStr.split(/\s+/));
  const intersection = [...expWords].filter(w => actWords.has(w));
  const union = new Set([...expWords, ...actWords]);
  
  return union.size > 0 ? intersection.length / union.size : 0;
}

/**
 * Calculate score for exercise arrays
 * @param {Array} expected - Expected exercises array
 * @param {Array} actual - Actual exercises array
 * @returns {number} Score (0-1) for exercise array match
 */
export function calculateExercisesScore(expected, actual) {
  // Handle missing exercises
  if (!expected && !actual) return 1; // Both null/undefined
  if (!expected || !actual) return 0; // One missing
  if (expected.length === 0 && actual.length === 0) return 1; // Both empty
  if (expected.length === 0 || actual.length === 0) return 0; // One empty
  
  // Score each exercise and average
  let totalScore = 0;
  const maxLength = Math.max(expected.length, actual.length);
  
  for (let i = 0; i < expected.length; i++) {
    const exp = expected[i];
    // Find best matching exercise in actual (by name)
    const matchingActual = actual.find(a => 
      a.name && exp.name && 
      (a.name.toLowerCase().includes(exp.name.toLowerCase()) ||
      exp.name.toLowerCase().includes(a.name.toLowerCase()))
    );
    
    if (matchingActual) {
      const exerciseScore = calculateSingleExerciseScore(exp, matchingActual);
      totalScore += exerciseScore;
    } else {
      // No matching exercise found
      totalScore += 0;
    }
  }
  
  return totalScore / maxLength;
}

/**
 * Calculate score for a single exercise comparison
 * @param {Object} expected - Expected exercise object
 * @param {Object} actual - Actual exercise object
 * @returns {number} Score (0-1) for single exercise match
 */
export function calculateSingleExerciseScore(expected, actual) {
  const scores = [];
  
  // Name match (most important, weighted 2x)
  scores.push(calculateStringMatch(expected.name, actual.name) * 2);
  
  // Numeric fields
  if (expected.sets !== undefined && actual.sets !== undefined) {
    scores.push(1 - calculateNormalizedError(expected.sets, actual.sets));
  }
  if (expected.reps !== undefined && actual.reps !== undefined) {
    scores.push(1 - calculateNormalizedError(expected.reps, actual.reps));
  }
  if (expected.weight !== undefined && actual.weight !== undefined) {
    scores.push(1 - calculateNormalizedError(expected.weight, actual.weight));
  }
  if (expected.durationMinutes !== undefined && actual.durationMinutes !== undefined) {
    scores.push(1 - calculateNormalizedError(expected.durationMinutes, actual.durationMinutes));
  }
  
  // Weight unit match
  if (expected.weightUnit !== undefined && actual.weightUnit !== undefined) {
    scores.push(calculateStringMatch(expected.weightUnit, actual.weightUnit));
  }
  
  return scores.length > 0 ? scores.reduce((sum, s) => sum + s, 0) / scores.length : 0;
}

/**
 * Compares two medication payloads field-by-field
 * Excludes notes field from scoring as it's free-text
 * @param {Object} expected - The expected payload from test dataset
 * @param {Object} actual - The actual payload from model response
 * @returns {Object} Detailed comparison results with scoring
 */
export function compareBiomarkerPayloads(expected, actual) {
  // Biomarkers are arrays of {name, value, unit}
  // Compare by matching biomarker names and checking if values and units match
  
  const comparison = {
    matches: [],
    mismatches: [],
    missing: [],
    extra: [],
    score: 0,
    totalBiomarkers: 0,
    matchPercentage: 0
  };

  // Handle null/undefined cases
  if (!expected || !actual) {
    comparison.totalBiomarkers = (expected?.length || 0) + (actual?.length || 0);
    if (expected) comparison.missing = expected;
    if (actual) comparison.extra = actual;
    return comparison;
  }

  // Convert arrays to maps keyed by biomarker name for easier comparison
  const expectedMap = new Map(expected.map(b => [b.name, b]));
  const actualMap = new Map(actual.map(b => [b.name, b]));

  // Get all unique biomarker names
  const allNames = new Set([...expectedMap.keys(), ...actualMap.keys()]);
  comparison.totalBiomarkers = allNames.size;

  allNames.forEach(name => {
    const expectedBio = expectedMap.get(name);
    const actualBio = actualMap.get(name);

    // Case 1: Biomarker only in actual (extra)
    if (!expectedBio) {
      comparison.extra.push(actualBio);
      return;
    }

    // Case 2: Biomarker only in expected (missing - bad!)
    if (!actualBio) {
      comparison.missing.push(expectedBio);
      return;
    }

    // Case 3: Biomarker exists in both - compare all fields
    const fieldsMatch = 
      expectedBio.name === actualBio.name &&
      expectedBio.value === actualBio.value &&
      expectedBio.unit === actualBio.unit;

    if (fieldsMatch) {
      comparison.matches.push(actualBio);
      comparison.score++;
    } else {
      comparison.mismatches.push({
        expected: expectedBio,
        actual: actualBio
      });
    }
  });

  // Calculate match percentage
  comparison.matchPercentage = comparison.totalBiomarkers > 0
    ? (comparison.score / comparison.totalBiomarkers) * 100
    : 0;

  return comparison;
}

export function compareMedicationPayloads(expected, actual) {
  // Fields to exclude from scoring (but still report)
  const EXCLUDED_FIELDS = ['notes'];
  
  const comparison = {
    matches: {},
    mismatches: {},
    missing: {},
    extra: {},
    score: 0,
    totalFields: 0,
    matchPercentage: 0,
    notesComparison: null  // Track notes separately for visibility
  };

  // Get all unique field names from both objects
  const allFields = new Set([
    ...Object.keys(expected || {}),
    ...Object.keys(actual || {})
  ]);

  allFields.forEach(field => {
    // Handle excluded fields separately
    if (EXCLUDED_FIELDS.includes(field)) {
      comparison.notesComparison = {
        expected: expected[field],
        actual: actual[field]
      };
      return; // Don't include in scoring
    }
    
    comparison.totalFields++;
    
    // Case 1: Field only exists in actual (extra field)
    if (!(field in expected)) {
      comparison.extra[field] = actual[field];
      return;
    }
    
    // Case 2: Field only exists in expected (missing field - this is bad!)
    if (!(field in actual)) {
      comparison.missing[field] = expected[field];
      return;
    }
    
    // Case 3: Field exists in both - compare values
    const expectedVal = expected[field];
    const actualVal = actual[field];
    
    if (JSON.stringify(expectedVal) === JSON.stringify(actualVal)) {
      comparison.matches[field] = actualVal;
      comparison.score++;
    } else {
      comparison.mismatches[field] = {
        expected: expectedVal,
        actual: actualVal
      };
    }
  });

  // Calculate match percentage (only on non-excluded fields)
  comparison.matchPercentage = comparison.totalFields > 0 
    ? (comparison.score / comparison.totalFields) * 100 
    : 0;

  return comparison;
}

export async function writeResultIncrementallyMeals(result, filePath, isbareRuntime = false) {
  if (!filePath) {
    throw new Error('File path is required');
  }

  let fs
  let path
  if (isbareRuntime) {
    fs = await import('bare-fs');
    path = await import('bare-path');
  } else {
    fs = await import('node:fs');
    path = await import('node:path');
  }

  // Load existing data
  let results = [];
  let counts = {
    truthy_payload: 0,
    falsy_payload: 0,
    truthy_error: 0,
    falsy_error: 0,
    parse_error: 0
  };

  if (fs.existsSync(filePath)) {
    const fileContent = fs.readFileSync(filePath, "utf-8");
    try {
      const data = JSON.parse(fileContent);
      results = data.results || [];
      if (data.summary) {
        counts = {
          truthy_payload: data.summary.truthy_payload || 0,
          falsy_payload: data.summary.falsy_payload || 0,
          truthy_error: data.summary.truthy_error || 0,
          falsy_error: data.summary.falsy_error || 0,
          parse_error: data.summary.parse_error || 0
        };
      }
    } catch {
      results = [];
    }
  }

  // Add new result and increment its count
  results.push(result);
  if (counts[result.classification] !== undefined) {
    counts[result.classification]++;
  }

  // Increment parse error count
  if (result.parseError) {
    counts.parse_error++;
  }

  // Calculate percentages
  const total = results.length;
  const payloadTotal = counts.truthy_payload + counts.falsy_payload;
  const errorTotal = counts.truthy_error + counts.falsy_error;

  const accuracy = total > 0 ? (counts.truthy_payload + counts.truthy_error) / total * 100 : 0;
  const payload_accuracy = payloadTotal > 0 ? counts.truthy_payload / payloadTotal * 100 : 0;
  const error_accuracy = errorTotal > 0 ? counts.truthy_error / errorTotal * 100 : 0;

  // Calculate average normalized error across all truthy_payload results
  const truthyPayloadResults = results.filter(r => r.classification === 'truthy_payload');
  let averageNormalizedError = 0;
  if (truthyPayloadResults.length > 0) {
    const sumNormalizedErrors = truthyPayloadResults.reduce((sum, r) => {
      return sum + (r.metrics?.averageNormalizedError || 0);
    }, 0);
    averageNormalizedError = sumNormalizedErrors / truthyPayloadResults.length;
  }

  // Write output
  const output = {
    summary: {
      ...counts,
      total,
      accuracy: Number(accuracy.toFixed(2)),
      payload_accuracy: Number(payload_accuracy.toFixed(2)),
      error_accuracy: Number(error_accuracy.toFixed(2)),
      average_normalized_error: Number(averageNormalizedError.toFixed(4))
    },
    results
  };

  // Ensure the directory exists before writing the file
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(filePath, JSON.stringify(output, null, 2));
}

export async function writeResultIncrementallyWorkouts(result, filePath, isbareRuntime = false) {
  if (!filePath) {
    throw new Error('File path is required');
  }

  let fs
  let path
  if (isbareRuntime) {
    fs = await import('bare-fs');
    path = await import('bare-path');
  } else {
    fs = await import('node:fs');
    path = await import('node:path');
  }

  // Load existing data
  let results = [];
  let counts = {
    truthy_payload: 0,
    falsy_payload: 0,
    truthy_error: 0,
    falsy_error: 0,
    parse_error: 0
  };

  if (fs.existsSync(filePath)) {
    const fileContent = fs.readFileSync(filePath, "utf-8");
    try {
      const data = JSON.parse(fileContent);
      results = data.results || [];
      if (data.summary) {
        counts = {
          truthy_payload: data.summary.truthy_payload || 0,
          falsy_payload: data.summary.falsy_payload || 0,
          truthy_error: data.summary.truthy_error || 0,
          falsy_error: data.summary.falsy_error || 0,
          parse_error: data.summary.parse_error || 0
        };
      }
    } catch {
      results = [];
    }
  }

  // Add new result and increment its count
  results.push(result);
  if (counts[result.classification] !== undefined) {
    counts[result.classification]++;
  }

  // Increment parse error count
  if (result.parseError) {
    counts.parse_error++;
  }

  // Calculate percentages
  const total = results.length;
  const payloadTotal = counts.truthy_payload + counts.falsy_payload;
  const errorTotal = counts.truthy_error + counts.falsy_error;

  const accuracy = total > 0 ? (counts.truthy_payload + counts.truthy_error) / total * 100 : 0;
  const payload_accuracy = payloadTotal > 0 ? counts.truthy_payload / payloadTotal * 100 : 0;
  const error_accuracy = errorTotal > 0 ? counts.truthy_error / errorTotal * 100 : 0;

  // Calculate average metrics across all truthy_payload results
  const truthyPayloadResults = results.filter(r => r.classification === 'truthy_payload');
  let averageNormalizedError = 0;
  let averageOverallScore = 0;
  let averageWorkoutTypeMatch = 0;
  let averageDescriptionMatch = 0;
  let averageIntensityMatch = 0;
  let averageExercisesScore = 0;
  
  if (truthyPayloadResults.length > 0) {
    const sumNormalizedErrors = truthyPayloadResults.reduce((sum, r) => {
      return sum + (r.metrics?.averageNormalizedError || 0);
    }, 0);
    averageNormalizedError = sumNormalizedErrors / truthyPayloadResults.length;

    const sumOverallScores = truthyPayloadResults.reduce((sum, r) => {
      return sum + (r.metrics?.overallScore || 0);
    }, 0);
    averageOverallScore = sumOverallScores / truthyPayloadResults.length;

    // Calculate average string matches (only count non-null values)
    let workoutTypeCount = 0, descCount = 0, intensityCount = 0, exercisesCount = 0;
    truthyPayloadResults.forEach(r => {
      if (r.metrics?.workoutTypeMatch !== null && r.metrics?.workoutTypeMatch !== undefined) {
        averageWorkoutTypeMatch += r.metrics.workoutTypeMatch;
        workoutTypeCount++;
      }
      if (r.metrics?.descriptionMatch !== null && r.metrics?.descriptionMatch !== undefined) {
        averageDescriptionMatch += r.metrics.descriptionMatch;
        descCount++;
      }
      if (r.metrics?.intensityLevelMatch !== null && r.metrics?.intensityLevelMatch !== undefined) {
        averageIntensityMatch += r.metrics.intensityLevelMatch;
        intensityCount++;
      }
      if (r.metrics?.exercisesScore !== null && r.metrics?.exercisesScore !== undefined) {
        averageExercisesScore += r.metrics.exercisesScore;
        exercisesCount++;
      }
    });
    
    averageWorkoutTypeMatch = workoutTypeCount > 0 ? averageWorkoutTypeMatch / workoutTypeCount : 0;
    averageDescriptionMatch = descCount > 0 ? averageDescriptionMatch / descCount : 0;
    averageIntensityMatch = intensityCount > 0 ? averageIntensityMatch / intensityCount : 0;
    averageExercisesScore = exercisesCount > 0 ? averageExercisesScore / exercisesCount : 0;
  }

  // Write output
  const output = {
    summary: {
      ...counts,
      total,
      accuracy: Number(accuracy.toFixed(2)),
      payload_accuracy: Number(payload_accuracy.toFixed(2)),
      error_accuracy: Number(error_accuracy.toFixed(2)),
      average_normalized_error: Number(averageNormalizedError.toFixed(4)),
      average_overall_score: Number(averageOverallScore.toFixed(4)),
      average_workout_type_match: Number(averageWorkoutTypeMatch.toFixed(4)),
      average_description_match: Number(averageDescriptionMatch.toFixed(4)),
      average_intensity_match: Number(averageIntensityMatch.toFixed(4)),
      average_exercises_score: Number(averageExercisesScore.toFixed(4))
    },
    results
  };

  // Ensure the directory exists before writing the file
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(filePath, JSON.stringify(output, null, 2));
}

export async function writeResultIncrementallyMedications(result, filePath, isbareRuntime = false) {
  if (!filePath) {
    throw new Error('File path is required');
  }

  let fs
  let path
  if (isbareRuntime) {
    fs = await import('bare-fs');
    path = await import('bare-path');
  } else {
    fs = await import('node:fs');
    path = await import('node:path');
  }

  // Load existing data
  let results = [];
  let counts = {
    truthy_payload: 0,
    partial_payload: 0,
    falsy_payload: 0,
    truthy_error: 0,
    falsy_error: 0,
    parse_error: 0
  };

  if (fs.existsSync(filePath)) {
    const fileContent = fs.readFileSync(filePath, "utf-8");
    try {
      const data = JSON.parse(fileContent);
      results = data.results || [];
      if (data.summary) {
        counts = {
          truthy_payload: data.summary.truthy_payload || 0,
          partial_payload: data.summary.partial_payload || 0,
          falsy_payload: data.summary.falsy_payload || 0,
          truthy_error: data.summary.truthy_error || 0,
          falsy_error: data.summary.falsy_error || 0,
          parse_error: data.summary.parse_error || 0
        };
      }
    } catch {
      results = [];
    }
  }

  // Add new result and increment its count
  results.push(result);
  if (counts[result.classification] !== undefined) {
    counts[result.classification]++;
  }

  // Increment parse error count
  if (result.parseError) {
    counts.parse_error++;
  }

  // Calculate percentages
  const total = results.length;
  const payloadTotal = counts.truthy_payload + counts.partial_payload + counts.falsy_payload;
  const errorTotal = counts.truthy_error + counts.falsy_error;

  const accuracy = total > 0 ? (counts.truthy_payload + counts.truthy_error) / total * 100 : 0;
  const payload_accuracy = payloadTotal > 0 ? counts.truthy_payload / payloadTotal * 100 : 0;
  const error_accuracy = errorTotal > 0 ? counts.truthy_error / errorTotal * 100 : 0;
  
  // Calculate average match percentage across all payload results
  const payloadResults = results.filter(r => 
    r.classification === 'truthy_payload' || 
    r.classification === 'partial_payload'
  );
  let averageMatchPercentage = 0;
  if (payloadResults.length > 0) {
    const sumMatchPercentages = payloadResults.reduce((sum, r) => {
      return sum + (r.payloadComparison?.matchPercentage || 0);
    }, 0);
    averageMatchPercentage = sumMatchPercentages / payloadResults.length;
  }

  // Write output
  const output = {
    summary: {
      ...counts,
      total,
      accuracy: Number(accuracy.toFixed(2)),
      payload_accuracy: Number(payload_accuracy.toFixed(2)),
      error_accuracy: Number(error_accuracy.toFixed(2)),
      average_match_percentage: Number(averageMatchPercentage.toFixed(2))
    },
    results
  };

  // Ensure the directory exists before writing the file
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(filePath, JSON.stringify(output, null, 2));
}

export async function writeResultIncrementallyBiomarkers(result, filePath, isbareRuntime = false) {
  if (!filePath) {
    throw new Error('File path is required');
  }

  let fs
  let path
  if (isbareRuntime) {
    fs = await import('bare-fs');
    path = await import('bare-path');
  } else {
    fs = await import('node:fs');
    path = await import('node:path');
  }

  let results = [];
  let counts = {
    truthy_payload: 0,
    partial_payload: 0,
    falsy_payload: 0,
    truthy_error: 0,
    falsy_error: 0,
    parse_error: 0,
  };

  // Read existing file, extract results and counts from summary if available
  if (fs.existsSync(filePath)) {
    const content = fs.readFileSync(filePath, 'utf8');
    if (content.trim()) {
      try {
        const parsed = JSON.parse(content);
        if (Array.isArray(parsed.results)) {
          results = parsed.results;
        }
        if (parsed.summary) {
          counts = {
            truthy_payload: parsed.summary.truthy_payload || 0,
            partial_payload: parsed.summary.partial_payload || 0,
            falsy_payload: parsed.summary.falsy_payload || 0,
            truthy_error: parsed.summary.truthy_error || 0,
            falsy_error: parsed.summary.falsy_error || 0,
            parse_error: parsed.summary.parse_error || 0
          };
        }
      } catch {
        results = [];
      }
    }
  }

  // Add the new result
  results.push(result);

  // Incrementally update counts
  if (counts[result.classification] !== undefined) {
    counts[result.classification]++;
  }

  // Increment parse error count
  if (result.parseError) {
    counts.parse_error++;
  }

  // Calculate statistics
  const total = results.length;
  const payloadTotal = counts.truthy_payload + counts.partial_payload + counts.falsy_payload;
  const errorTotal = counts.truthy_error + counts.falsy_error;

  const accuracy = total > 0
    ? ((counts.truthy_payload + counts.truthy_error) / total) * 100
    : 0;

  const payload_accuracy = payloadTotal > 0
    ? (counts.truthy_payload / payloadTotal) * 100
    : 0;

  const error_accuracy = errorTotal > 0
    ? (counts.truthy_error / errorTotal) * 100
    : 0;

  // Calculate average match percentage across all results that expect payload
  const allPayloadResults = results.filter(r => 
    r.classification === 'truthy_payload' || 
    r.classification === 'partial_payload' || 
    r.classification === 'falsy_payload'
  );
  let averageMatchPercentage = 0;
  if (allPayloadResults.length > 0) {
    const sumMatchPercentages = allPayloadResults.reduce((sum, r) => {
      return sum + (r.biomarkerComparison?.matchPercentage || 0);
    }, 0);
    averageMatchPercentage = sumMatchPercentages / allPayloadResults.length;
  }

  const output = {
    summary: {
      ...counts,
      total,
      accuracy: Number(accuracy.toFixed(2)),
      payload_accuracy: Number(payload_accuracy.toFixed(2)),
      error_accuracy: Number(error_accuracy.toFixed(2)),
      average_match_percentage: Number(averageMatchPercentage.toFixed(2)),
    },
    results,
  };

  // Ensure the directory exists before writing the file
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(filePath, JSON.stringify(output, null, 2));
}

/**
 * Compares two symptom payloads field-by-field
 * Symptom schema: {name: string, description: string, severity?: "mild" | "moderate" | "severe"}
 * @param {Object} expected - The expected payload from test dataset
 * @param {Object} actual - The actual payload from model response
 * @returns {Object} Detailed comparison results with scoring
 */
export function compareSymptomPayloads(expected, actual) {
  // Simple comparison: only check if the name is semantically correct
  if (!expected?.name || !actual?.name) {
    return { 
      expectedName: expected?.name || null,
      actualName: actual?.name || null,
      isMatch: false, 
      matchPercentage: 0 
    };
  }

  const expectedName = expected.name.toLowerCase().trim();
  const actualName = actual.name.toLowerCase().trim();

  // Exact or substring match for semantic similarity
  const isMatch = expectedName === actualName || 
                  actualName.includes(expectedName) || 
                  expectedName.includes(actualName);

  return {
    expectedName: expected.name,
    actualName: actual.name,
    isMatch,
    matchPercentage: isMatch ? 100 : 0
  };
}

export async function writeResultIncrementallySymptoms(result, filePath, isbareRuntime = false) {
  if (!filePath) {
    throw new Error('File path is required');
  }

  let fs
  let path
  if (isbareRuntime) {
    fs = await import('bare-fs');
    path = await import('bare-path');
  } else {
    fs = await import('node:fs');
    path = await import('node:path');
  }

  let results = [];
  let counts = {
    truthy_payload: 0,
    falsy_payload: 0,
    truthy_error: 0,
    falsy_error: 0,
    parse_error: 0,
  };

  // Read existing file, extract results and counts from summary if available
  if (fs.existsSync(filePath)) {
    const content = fs.readFileSync(filePath, 'utf8');
    if (content.trim()) {
      try {
        const parsed = JSON.parse(content);
        if (Array.isArray(parsed.results)) {
          results = parsed.results;
        }
        if (parsed.summary) {
          counts = {
            truthy_payload: parsed.summary.truthy_payload || 0,
            falsy_payload: parsed.summary.falsy_payload || 0,
            truthy_error: parsed.summary.truthy_error || 0,
            falsy_error: parsed.summary.falsy_error || 0,
            parse_error: parsed.summary.parse_error || 0
          };
        }
      } catch {
        results = [];
      }
    }
  }

  // Add the new result
  results.push(result);

  // Incrementally update counts
  if (counts[result.classification] !== undefined) {
    counts[result.classification]++;
  }

  // Increment parse error count
  if (result.parseError) {
    counts.parse_error++;
  }

  // Calculate statistics
  const total = results.length;
  const payloadTotal = counts.truthy_payload + counts.falsy_payload;
  const errorTotal = counts.truthy_error + counts.falsy_error;

  const accuracy = total > 0
    ? ((counts.truthy_payload + counts.truthy_error) / total) * 100
    : 0;

  const payload_accuracy = payloadTotal > 0
    ? (counts.truthy_payload / payloadTotal) * 100
    : 0;

  const error_accuracy = errorTotal > 0
    ? (counts.truthy_error / errorTotal) * 100
    : 0;

  const output = {
    summary: {
      ...counts,
      total,
      accuracy: Number(accuracy.toFixed(2)),
      payload_accuracy: Number(payload_accuracy.toFixed(2)),
      error_accuracy: Number(error_accuracy.toFixed(2)),
    },
    results,
  };

  // Ensure the directory exists before writing the file
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(filePath, JSON.stringify(output, null, 2));
}



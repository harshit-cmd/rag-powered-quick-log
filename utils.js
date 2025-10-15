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



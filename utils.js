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

export async function writeResultIncrementally(result, filePath, isbareRuntime = false) {
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
  } else if (result.parseError) {
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
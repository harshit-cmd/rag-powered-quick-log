#!/usr/bin/env node
/**
 * Script to summarize benchmark results and calculate average accuracy.
 * Reads all JSON files from the specified benchmark results folder.
 *
 * Usage:
 *     node summarize_benchmarks.js [folder_path]
 *
 * Examples:
 *     node summarize_benchmarks.js benchmark-results/sqlite-rag-powered
 *     node summarize_benchmarks.js benchmark-results/hyperdb-rag-powered-impl
 *     node summarize_benchmarks.js benchmark-results/current
 *
 * If no folder is specified, defaults to benchmark-results/sqlite-rag-powered/.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Read all JSON benchmark files from the specified folder.
 *
 * @param {string} folderPath - Path to the folder containing benchmark JSON files
 * @returns {Array<Object>} List of summaries from each benchmark file
 */
export function readBenchmarkFiles(folderPath) {
    const summaries = [];

    // Get all JSON files
    const files = fs.readdirSync(folderPath);
    const jsonFiles = files
        .filter(file => file.endsWith('.json'))
        .sort();

    for (const fileName of jsonFiles) {
        const filePath = path.join(folderPath, fileName);
        try {
            const fileContent = fs.readFileSync(filePath, 'utf8');
            const data = JSON.parse(fileContent);

            if (data.summary) {
                const summary = { ...data.summary };
                summary.filename = fileName;
                summaries.push(summary);
            }
        } catch (error) {
            console.log(`Error reading ${fileName}: ${error.message}`);
        }
    }

    return summaries;
}

/**
 * Calculate average metrics across all summaries.
 *
 * @param {Array<Object>} summaries - List of benchmark summaries
 * @returns {Object} Object containing averaged metrics
 */
export function calculateAverages(summaries) {
    if (summaries.length === 0) {
        return {};
    }

    // Metrics to average
    const metrics = [
        'truthy_payload', 'falsy_payload', 'truthy_error', 'falsy_error',
        'parse_error', 'total', 'accuracy', 'payload_accuracy', 'error_accuracy',
        'average_normalized_error'
    ];

    const averages = {};

    for (const metric of metrics) {
        const values = summaries
            .filter(s => metric in s)
            .map(s => s[metric] || 0);

        if (values.length > 0) {
            averages[metric] = values.reduce((sum, val) => sum + val, 0) / values.length;
        }
    }

    return averages;
}

/**
 * Print formatted summary of all benchmark results.
 *
 * @param {Array<Object>} summaries - List of benchmark summaries
 * @param {Object} averages - Averaged metrics
 */
export function printSummary(summaries, averages) {
    console.log('='.repeat(80));
    console.log(`BENCHMARK SUMMARY - ${summaries.length} files analyzed`);
    console.log('='.repeat(80));
    console.log();

    // Print individual file summaries
    console.log('Individual File Results:');
    console.log('-'.repeat(80));

    summaries.forEach((summary, index) => {
        console.log(`\n${index + 1}. ${summary.filename}`);
        console.log(`   Total Tests: ${summary.total || 0}`);
        console.log(`   Accuracy: ${(summary.accuracy || 0).toFixed(2)}%`);
        console.log(`   Payload Accuracy: ${(summary.payload_accuracy || 0).toFixed(2)}%`);
        console.log(`   Error Accuracy: ${(summary.error_accuracy || 0).toFixed(2)}%`);
        console.log(`   Average Normalized Error: ${(summary.average_normalized_error || 0).toFixed(4)}`);
        console.log(`   Truthy Payload: ${summary.truthy_payload || 0}`);
        console.log(`   Falsy Payload: ${summary.falsy_payload || 0}`);
        console.log(`   Truthy Error: ${summary.truthy_error || 0}`);
        console.log(`   Falsy Error: ${summary.falsy_error || 0}`);
        console.log(`   Parse Error: ${summary.parse_error || 0}`);
    });

    // Print averages
    console.log('\n' + '='.repeat(80));
    console.log('AVERAGE METRICS ACROSS ALL FILES:');
    console.log('='.repeat(80));
    console.log(`Average Total Tests: ${(averages.total || 0).toFixed(2)}`);
    console.log(`Average Accuracy: ${(averages.accuracy || 0).toFixed(2)}%`);
    console.log(`Average Payload Accuracy: ${(averages.payload_accuracy || 0).toFixed(2)}%`);
    console.log(`Average Error Accuracy: ${(averages.error_accuracy || 0).toFixed(2)}%`);
    console.log(`Average Normalized Error (truthy_payload only): ${(averages.average_normalized_error || 0).toFixed(4)}`);
    console.log();
    console.log(`Average Truthy Payload: ${(averages.truthy_payload || 0).toFixed(2)}`);
    console.log(`Average Falsy Payload: ${(averages.falsy_payload || 0).toFixed(2)}`);
    console.log(`Average Truthy Error: ${(averages.truthy_error || 0).toFixed(2)}`);
    console.log(`Average Falsy Error: ${(averages.falsy_error || 0).toFixed(2)}`);
    console.log(`Average Parse Error: ${(averages.parse_error || 0).toFixed(2)}`);
    console.log('='.repeat(80));
}

function getScriptDir() {
    // __dirname replacement for ESM
    const __filename = fileURLToPath(import.meta.url);
    return path.dirname(__filename);
}

export function main() {
    // Get folder path from command line argument or use default
    const scriptDir = getScriptDir();

    let benchmarkFolder;

    if (process.argv.length > 2) {
        // Use provided folder path (resolve relative to the process current working directory)
        const folderArg = process.argv[2];
        benchmarkFolder = path.isAbsolute(folderArg)
            ? folderArg
            : path.resolve(process.cwd(), folderArg);
    } else {
        // Default folder
        benchmarkFolder = path.join(scriptDir, 'benchmark-results', 'sqlite-rag-powered');
    }

    if (!fs.existsSync(benchmarkFolder)) {
        console.log(`Error: Folder not found: ${benchmarkFolder}`);
        console.log(`\nUsage: node ${path.basename(process.argv[1])} [folder_path]`);
        console.log(`Example: node ${path.basename(process.argv[1])} benchmark-results/current`);
        return;
    }

    console.log(`Reading benchmark files from: ${benchmarkFolder}`);
    console.log();

    // Read all benchmark files
    const summaries = readBenchmarkFiles(benchmarkFolder);

    if (summaries.length === 0) {
        console.log('No benchmark files found!');
        return;
    }

    // Calculate averages
    const averages = calculateAverages(summaries);

    // Print results
    printSummary(summaries, averages);
}

// ESM entrypoint (Node >= 14)
// Only run main if this is called as a CLI, not imported.
if (import.meta.url === process.argv[1] || import.meta.url === `file://${process.argv[1]}`) {
    main();
}

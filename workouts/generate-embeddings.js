import {
  loadModel,
  GTE_LARGE_FP16,
  embed,
} from "@tetherto/qvac-sdk";
import fs from "fs";
import workoutRagDataset from "./workout-rag-dataset-cleaned.json" with { type: "json" };

let embeddingModelId;

const initEmbeddingModel = async () => {
  console.log("Loading embedding model...");
  embeddingModelId = await loadModel({
    modelSrc: GTE_LARGE_FP16,
    modelType: "embeddings",
    onProgress: (progress) => {
      process.stdout.write(`\rLoading embedding model... ${progress.percentage.toFixed(4)}%`);
    },
  });
  console.log("\n✅ Embedding model loaded!");
};

const main = async () => {
  await initEmbeddingModel();

  const dataWithEmbeddings = [];
  const total = workoutRagDataset.length;

  console.log(`\n📊 Processing ${total} entries...`);

  for (let i = 0; i < workoutRagDataset.length; i++) {
    const sample = workoutRagDataset[i];
    
    // Generate embedding for the prompt
    const embedding = await embed({ 
      modelId: embeddingModelId, 
      text: sample.prompt 
    });

    // Add embedding to the sample
    const sampleWithEmbedding = {
      ...sample,
      embedding: embedding,
    };

    dataWithEmbeddings.push(sampleWithEmbedding);

    // Progress indicator
    if ((i + 1) % 50 === 0 || i === total - 1) {
      const progress = ((i + 1) / total * 100).toFixed(2);
      console.log(`✨ Embedded ${i + 1}/${total} entries (${progress}%)`);
    }
  }

  // Save to file
  const outputPath = "./workouts/workouts-datasets/workout-rag-dataset-cleaned-with-embeddings.json";
  console.log(`\n💾 Saving to ${outputPath}...`);
  
  fs.writeFileSync(
    outputPath,
    JSON.stringify(dataWithEmbeddings, null, 2)
  );

  console.log("✅ Successfully created workout-rag-dataset-cleaned-with-embeddings.json!");
  console.log(`📊 Total entries with embeddings: ${dataWithEmbeddings.length}`);
  console.log(`📉 Removed from original: ${1000 - dataWithEmbeddings.length} entries`);

  process.kill(process.pid);
};

main().catch((error) => {
  console.error("❌ Error:", error);
  process.exit(1);
});


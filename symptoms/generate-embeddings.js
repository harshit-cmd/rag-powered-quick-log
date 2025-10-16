import {
  loadModel,
  GTE_LARGE_FP16,
  embed,
} from "@tetherto/qvac-sdk";
import fs from "fs";
import comprehensiveSeedData from "./symptoms-datasets/comprehensive-seed-data.json" with { type: "json" };
// import symptomsTestDataset from "./symptoms-datasets/test-dataset.json" with { type: "json" };


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
  const total = [...comprehensiveSeedData].length;

  console.log(`\n📊 Processing ${total} entries...`);

    for (let i = 0; i < [...comprehensiveSeedData].length; i++) {
    const sample = [...comprehensiveSeedData][i];
    
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
      console.log(`\r✨ Embedded ${i + 1}/${total} entries (${progress}%)`);
    }
  }

  // Save to file
  const outputPath = "./symptoms/symptoms-datasets/comprehensive-seed-data-with-embeddings.json";
  console.log(`\n💾 Saving to ${outputPath}...`);
  
  fs.writeFileSync(
    outputPath,
    JSON.stringify(dataWithEmbeddings, null, 2)
  );

  console.log("✅ Successfully created comprehensive-seed-data-with-embeddings.json!");
  console.log(`📊 Total entries with embeddings: ${dataWithEmbeddings.length}`);

  process.kill(process.pid);
};

main().catch((error) => {
  console.error("❌ Error:", error);
  process.exit(1);
});


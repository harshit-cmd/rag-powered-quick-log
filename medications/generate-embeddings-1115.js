import {
  loadModel,
  GTE_LARGE_FP16,
  embed,
} from "@tetherto/qvac-sdk";
import fs from 'fs';

const ragDataset = JSON.parse(
  fs.readFileSync('./medications/medications-datasets/rag-dataset-1115.json', 'utf8')
);

let embeddingModelId;

const initEmbeddingModel = async () => {
  embeddingModelId = await loadModel({
    modelSrc: GTE_LARGE_FP16,
    modelType: "embeddings",
    onProgress: (progress) => {
      process.stdout.write(`\rLoading embedding model... ${progress.percentage.toFixed(4)}%`);
    },
  });
  console.log('\nEmbedding model loaded!');
};

const main = async () => {
  await initEmbeddingModel();
  
  const datasetWithEmbeddings = [];
  let processed = 0;
  
  for (const sample of ragDataset) {
    const embedding = await embed({ 
      modelId: embeddingModelId, 
      text: sample.prompt 
    });
    
    datasetWithEmbeddings.push({
      ...sample,
      embedding: embedding
    });
    
    processed++;
    process.stdout.write(`\rProcessed ${processed}/${ragDataset.length} embeddings`);
  }
  
  console.log('\n\nWriting dataset with embeddings...');
  fs.writeFileSync(
    './medications/medications-datasets/rag-dataset-1115-with-embeddings.json',
    JSON.stringify(datasetWithEmbeddings, null, 2)
  );
  
  console.log('Done! Created rag-dataset-1115-with-embeddings.json');
  process.exit(0);
};

main().catch(console.error);



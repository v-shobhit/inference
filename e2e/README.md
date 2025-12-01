\# E2E: RAG benchmark
This is a WIP proposal, and will undergo changes.

## Benchmark flow
We start with a corpus of documents and a set of user queries. 

First, we preprocess the corpus to get a vector DB.

Then, given a user query, we perfrom:
1. retrieval
2. reranking
3. answer generation

## Preliminary: Environment setup
The recommended environment setup depends on enroot installation - however, it is simple enough that docker may be used. 

### Base environment
It is assumed that the user starts from a Ubuntu PyTorch base image (sandbox/container)

You may use either:
- [Ubuntu base image from dockerhub](https://hub.docker.com/_/ubuntu) (not recommended)
- [NVIDIA PyTorch image from NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) (recommended)
- [AMD ROCm/PyTorch image from dockerhub](https://hub.docker.com/r/rocm/pytorch/tags) (not tested yet)

### Enroot setup
If you have enroot, you can get started by:
```bash
mkdir containers/ && cd containers \
  enroot import -o nv-pyt-25.10.sqsh dockerd://nvcr.io/nvidia/pytorch:25.10-py3 && cd -
enroot create --name pyt containers/nv-pyt-25.10.sqsh
enroot start --root --rw \
  --mount $(pwd):/work pyt

## Once inside sandbox, install dependancies via
cd /work && ./setup.sh
```

## Step 1: Download documents

The first step is to download the frames dataset, then the documents referred to in this dataset. 

```bash
./download_frames.sh  # downloads patched frames dataset from huggingface

python3 wikifetch/download_frames_docs.py \
  --tsv-path data/frames/test.tsv \
  --output-dir wiki_articles --workers 16
# Downloads the wikipedia content from the article urls in the frames dataset
```

## Step 2: Chunk the articles into passages
Due to the context length limitations of the retriever (embedding) and reranker models, we chunk the articles into smaller passages. 
```bash
python chunk_frames_wiki.py \
  --articles-dir wiki_articles \
  --output wiki_articles/passages.json \
  --similarity-threshold 0.25 \
  --chunk-size 512 \
  --overlap 10 \
  --workers 8 \
  --device cuda
```

## Step 3: Create a VectorDB from passages
Once the set of articles are transformed into passages, we can create a vector DB for lookup.

```bash
python create_vector_store.py \
  --passages wiki_articles/passages.json \
  --output wiki_articles/passages_vectordb.pkl \
  --device cuda
```

## Step 4: Run retrieval component on FRAMES

Now, we have the vectorDB - in order to run the retriever component, we first create a `config.yml` file:

```yaml
data:
  vector_store: "wiki_articles/passages_vectordb.pkl"
  tsv_path: "data/frames-benchmark/test.tsv"
  num_prompts: all

device: "cuda"

retrieval:
  model: "intfloat/e5-base-v2"
  top_k: 20 # top number of docs retrieved

reranker:
  enabled: true
  model: "colbert-ir/colbertv2.0"
  top_p: 0.4 # cumulative score of top ranked docs that are taken

rewriter:
  enabled: true
  endpoint: "http://localhost:30001/v1" # SGLang endpoint
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  steps: 5 # number of times rewriter is invoked
  queries_per_step: 5 # number of queries generated per step
  temperature: 0.7 # LLM sampling
  max_tokens: 1024 # LLM sampling

parallel:
  max_workers: 4  # Optional: Number of parallel workers (default: 1 = sequential)

output:
  results: "data/retrieval_results.pkl"    # Results of retrieval pipeline
  rewriter_io: "data/rewriter_io.pkl"      # Optional: save query generation I/O
  retriever_io: "data/retriever_io.pkl"    # Optional: save retrieval I/O
  reranker_io: "data/reranker_io.pkl"      # Optional: save reranking I/O
```

Then, 
```bash
python3 run_frames_retrieval.py --config config.yml
```
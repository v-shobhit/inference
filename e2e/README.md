# E2E: RAG benchmark
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

## Corpus creation
Starting from [FRAMES](https://huggingface.co/datasets/google/frames-benchmark), we have a set of tuples as:
```
[UserQuery, WikiLinks, Answer]
```

### Method 1: Clean Wikipedia Content (Recommended)
We extract all the unique Wikipedia links and download **clean article content** directly using the Wikipedia API. This approach is inspired by [WikiExtractor](https://github.com/attardi/wikiextractor) and provides several advantages over HTML/PDF:

**Advantages:**
- ✅ **Cleaner content**: No navigation bars, sidebars, references, or irrelevant HTML artifacts
- ✅ **Faster processing**: Direct API access, no HTML/PDF parsing needed
- ✅ **Better RAG quality**: Only article text content, improving retrieval accuracy
- ✅ **Automatic cleaning**: Removes "See also", "References", "External links" sections
- ✅ **Smaller storage**: Plain text files instead of HTML/PDF

You may use the [frames_wiki_fetch.py](./frames_wiki_fetch.py) script:
```bash
$ python3 frames_wiki_fetch.py --help
usage: frames_wiki_fetch.py [-h] [--tsv-path TSV_PATH] [--output-dir OUTPUT_DIR]
                             [--max-urls MAX_URLS] [--download-workers N]
                             [--create-chunks] [--chunk-size CHUNK_SIZE] 
                             [--overlap OVERLAP] [--semantic] [--no-semantic]
                             [--chunk-workers N] [--chunk-device {cpu,cuda}]

Download clean Wikipedia articles from FRAMES dataset using Wikipedia API

options:
  -h, --help            show this help message and exit
  --tsv-path TSV_PATH   Input TSV file with FRAMES data (default: download from Hugging Face)
  --output-dir OUTPUT_DIR
                        Output directory for clean article text files (default: wiki_clean_articles)
  --max-urls MAX_URLS   Maximum number of URLs to process (default: all)
  --download-workers N  Number of parallel download workers (default: 10, max: 20)
  --create-chunks       Create passages JSON file with chunked content
  --chunk-size CHUNK_SIZE
                        Maximum characters per chunk (default: 512)
  --overlap OVERLAP     Overlap between chunks in characters (default: 50)
  --semantic            Use semantic chunking (default: True)
  --no-semantic         Disable semantic chunking
  --chunk-workers N     Parallel chunk workers (default: 1, max 8 for semantic)
  --chunk-device        Device for chunking: cpu or cuda (default: auto)

## Sample usage - Download and create passages in one step
$ python3 frames_wiki_fetch.py --output-dir wiki_clean --download-workers 20 --create-chunks --chunk-size 512 --overlap 50

## Or download first, then create passages later
$ python3 frames_wiki_fetch.py --output-dir wiki_clean --download-workers 20
$ python3 frames_wiki_fetch.py --output-dir wiki_clean --create-chunks
```

### Method 2: HTML/PDF Download (Legacy)
**Note:** This method is deprecated in favor of the clean Wikipedia content approach above.

The legacy approach downloads web pages as PDFs using [`wkhtmltopdf`](https://wkhtmltopdf.org/) or as HTML files.

You may use the [download.py](./download.py) script (formerly download_pdf.py):
```bash
$ python3 download.py --help
usage: download.py [-h] [--tsv_path TSV_PATH] [--max_urls MAX_URLS]
                   [--output_dir OUTPUT_DIR] [--output_data OUTPUT_DATA]
                   [--processes PROCESSES] [--format {pdf,html}]

Download FRAMES dataset from Hugging Face and convert URLs to PDFs or HTML files

options:
  -h, --help            show this help message and exit
  --tsv_path TSV_PATH   Input TSV file (default: download FRAMES dataset)
  --max_urls MAX_URLS   Maximum number of URLs to process (default: all)
  --output_dir OUTPUT_DIR
                        Output directory for downloaded files (default: doc_downloads)
  --output_data OUTPUT_DATA
                        Output directory for dataset file (default: data)
  --processes PROCESSES
                        Number of parallel processes (default: 10)
  --format {pdf,html}   Output format: pdf or html (default: pdf)

## Sample usage
$ python3 download.py --output_dir doc_pdf --format pdf --processes 30
$ python3 download.py --output_dir doc_html --format html --processes 30
```

## Corpus preprocessing

### For Clean Wikipedia Content (Method 1)
When using the `frames_wiki_fetch.py` script with the `--create-chunks` flag, preprocessing is **already done**! The script outputs:
- Clean text files (one per Wikipedia article)
- Metadata JSON files (article title, URL, etc.)
- A `passages.json` file with chunked content ready for RAG

The passages JSON has the following schema:
```json
{
    "index": 0,
    "article_filename": "Article_Title.txt",
    "article_title": "Article Title",
    "article_url": "https://en.wikipedia.org/wiki/Article_Title",
    "source_url": "https://en.wikipedia.org/wiki/Article_Title",
    "passage": "Clean text passage from the article",
    "passage_length": 450
}
```

**No additional preprocessing needed!** The text is already clean and chunked.

### For HTML Files (Method 2 - Legacy)
If you downloaded HTML files, you can process them using the [preprocess_html.py](./preprocess_html.py) script:
```bash
$ python3 preprocess_html.py --help
# Process all HTML files in a folder
python preprocess_html.py path/to/html/folder --output path/to/output/folder --workers 8
```

### For PDF Files (Method 2 - Legacy)
If you downloaded PDFs, you need to extract text content using [PyMuPDF](https://pypi.org/project/PyMuPDF/) (`fitz`).

Important considerations:
- Extracted text from a single PDF document has many characters.
- However, embedding models (especially rerankers like [`ColBERTv2`](https://huggingface.co/colbert-ir/colbertv2.0)) have a size restriction on the maximum length of sequence they can encode (~512). 
- Thus, we break down PDFs into chunks called "passages".
    - Each passage belongs to a single unique PDF source.

In this step:
- Input: a set of PDFs
- Output:
    - a set of txt documents (1 per PDF), and
    - a JSON file of passages.

The passages JSON has a schema as follows:
```json
{
    "index": 0,
    "pdf_filename": "name_of_file.pdf",
    "passage": "Long passage from (part of) pdf_filename"
}
```

You may use the [`read_pdf.py`](./read_pdf.py) script:
```bash
$ python3 read_pdf.py --help
usage: read_pdf.py [-h] [--json-file JSON_FILE] [--max-files MAX_FILES] [--max-length MAX_LENGTH]
                   [--overlap OVERLAP]
                   input_dir output_dir

Extract text from all PDFs in a directory

positional arguments:
  input_dir             Input directory containing PDF files
  output_dir            Output directory for text files

options:
  -h, --help            show this help message and exit
  --json-file JSON_FILE
                        Output JSON file path for passages data (enables JSON creation)
  --max-files MAX_FILES
                        Maximum number of PDF files to process (default: all files)
  --max-length MAX_LENGTH
                        Maximum length of each passage in characters (default: 512)
  --overlap OVERLAP     Overlap between passages in characters (default: 50)

## Sample usage
$ python3 read_pdf.py doc_pdf doc_txt_len256_overlap32 --max-length 256 --json-file doc_txt_fixed_len256_overlap32/passages.json --overlap 32
```

### Important Notes on Chunking:
1. **Clean Wikipedia method**: Uses sentence-aware chunking that preserves sentence boundaries for better context.
2. **PDF/HTML method**: Character-level chunking may split sentences mid-way.
3. Passage size directly affects vector operations. Consider the impact of passage length + overlap on vector size, ingestion time, and lookup time.
4. For the clean Wikipedia method, recommended settings: `--chunk-size 512 --overlap 50`

## Single-shot lookup
1. Embed query: `Query text` -> `Query Tokens` -> `Query vector`
2. Perform vector similarity search, and return top-k documents. 
3. Perform reranking using ColBERT (Late interaction and `MaxSim` scoring)

Rerankers: Slow but accurate  
Retrievers: Fast but less accurate

## Multi-step lookup (TODO)
Instead of a single step retrieval, we perform multiple steps. In each step, we give the LLM partial retrieved context, and the user query - and ask it to generate search queries. This helps in breaking down multi-step reasoning questions.

Consider, as an example, the below query: 
```none
Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?
```

This is a classic multi-step reasoning. The logical deduction of a well-performing system is: 
```
- What year did New York City FC win their first MLS Cup title
(retrieve docs regarding MLS cup winners)
(say, answer is 2005)
- Who won the French Open Mens Singles tournament in 2005?
(retrieve 2005 French open document)
```

The flow now looks something like: 
1. User query comes in
2. Repeat 1..n times:  
    1. Given to query rewriter, which gives at most k search queries.
    2. For each query:
        1. Encode into vector
        2. Perform vector search and retrieve relevant documents
3. Rerank retrieved documents, and choose top-n (call this filtered documents)
4. Give filtered documents + user query to LLM generator

The query rewriter may be thought of as an LLM with the following prompt: 
```
You are an expert at generating search queries to help answer complex questions using a collection of Wikipedia articles. 

Given the following:
- The user's original question.
- Relevant facts or documents already gathered so far (if any).

Your task:  
Generate [k] concise, focused search queries that could be used to find specific information from Wikipedia to help answer the question.  
- Make each query target a different aspect of the problem or missing information.  
- Avoid duplicating information already in the context.  
- Do not reference source filenames, document titles, or include any special characters.
- Think step by step before writing each query.
- List the missing pieces of information, then write k queries that could best retrieve them.

[User Question:]
{user_question}

[Known Facts / Retrieved Documents:]
{summarized_partial_context}

```
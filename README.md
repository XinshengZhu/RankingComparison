# Ranking Comparison Among Three Search Systems

## Overview

This project is written and tested entirely on macOS Sequoia throughout the development process.
It may not work on other operating systems without modifications.

We perform ranking comparison among three systems:
1. **Standard BM25 Search System**: The modified standard BM25 search system from former Assignment #2, developed based on the concept of inverted index.
2. **HNSW-based Retrieval System**: The dense vector retrieval system using HNSW algorithm, built with vector embeddings, implemented with the professional library Faiss.
3. **Cascading Rerank System**: The cascading rerank system reranking the results from the standard BM25 search system by computing the cosine similarity of vector embeddings.

## Files

Here's a brief description of all code files of the project:

```
RankingComparison/
├─── generated_data/                # Directory for generated intermediate data and indexes
├─── provided_data/                 # Directory for provided input data like embeddings and relevance judgments
│
├─── SearchSystem_subset/           # Modified standard BM25 search system implementation, running on a subset of the collection dataset, written in C
│    ├─── DataParser/               # Data parsing components, sqlite3 database for displaying the results removed
│    │    └─── ...                  # Source files similar to Assignment #2, omitted for brevity
│    ├─── IndexBuilder/             # Index building components, linearly impact score compression applied
│    │    └─── ...                  # Source files similar to Assignment #2, omitted for brevity
│    ├─── QueryProcessor/           # Query processing components, generating the search results for a bunch of queries once into a file, linearly impact score decompression applied
│    │    └─── ...                  # Source files similar to Assignment #2, omitted for brevity
│    └─── CMakeLists.txt            # CMake build configuration file, totally not changed
│
├─── ann_hnsw.py                    # HNSW index implementation for dense retrieval, supports index building, saving, loading and k-NN search
├─── cascading_rerank.py            # Reranking implementation using semantic similarity, reranks BM25 results using embedding cosine similarity
├─── metrics.py                     # Information retrieval evaluation metrics (MRR, Recall, MAP, NDCG)
├─── standard_bm25.py               # Standard BM25 search system pipeline, handles data preparation and standard BM25 search system execution
└─── utils.py                       # Utility functions for data handling, includes data loading and formatting utilities
```

## Requirements

### Datasets

- [Collection of Passage Retrieval Dataset](https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz) is required to be downloaded and extracted necessarily using the following commands:

  ```bash
  $ curl -O https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
  $ tar -xzvf collection.tar.gz
  ```
  The `collection.tsv` file should be moved to the `provided_data` directory before running the Python scripts.


- [Queries of Passage Retrieval Dataset](https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz) is required to be downloaded and extracted into the `provided_data` dictionary necessarily using the following commands:

  ```bash
  $ curl -O https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
  $ tar -xzvf queries.tar.gz
  ```
    The `queries.dev.tsv` and `queries.eval.tsv` files should be moved to the `provided_data` directory before running the Python scripts.


- We are also provided with the following files for setup and evaluation through [Google Drive](https://drive.google.com/drive/u/0/folders/1Sc8XSsCOtq_gxQoEEec1HxkMSSGHsFgd), all of which should be moved to the `provided_data` directory before running the Python scripts:
  - `msmarco_passage_embeddings_subset.h5`: Contains the ids and embeddings of 1m passages out of 8.8m in the original dataset
  - `msmarco_dev_eval_embeddings.h5`: Contains the ids and embeddings of all queries that need to be evaluated in this assignment
  - `qrels.dev.tsv`: Contains the relevance judgments for the dev set
  - `qrels.eval.one.tsv`: Contains the relevance judgments for the eval set
  - `qrels.eval.two.tsv`: Contains the relevance judgments for the eval set

### Environments

- The `cmake` build system is required to compile our `SearchSystem_subset` part of Standard BM25 Search System. Install it using the following command:

  ```bash
  $ brew install cmake
  ```

- Several Python packages are required to run our Python scripts. Install them using the following commands:

  ```bash
  # For HNSW index building and searching
  $ pip install faiss-cpu
  # For h5 file reading
  $ pip install h5py
  # For basic data manipulation
  $ pip install numpy
  # For computing cosine similarity
  $ pip install scikit-learn
  ```

## Compilation for Standard BM25 Search System Written in C

* Move to the `RankingComparison/SearchSystem_subset` dictionary

  ```bash
  $ cd RankingComparison/SearchSystem_subset/
  ```

* Create a build directory and move into it

  ```bash
  $ mkdir build && cd build
  ```

* Run CMake to generate the build files

  ```bash
  $ cmake ..
  ```

* Build the project using the generated build files

  ```bash
  $ cmake --build .
  ```
  
* Move back to the root `RankingComparison` directory after compilation

  ```bash
  $ cd ../../
  ```

It should be noted that the `DataParser`, `IndexBuilder`, and `QueryProcessor` executables will be generated in the `RankingComparison/SearchSystem_subset/build` directory.
All needed file paths are hardcoded in the C source code and our Python scripts can execute these executables automatically in the root `RankingComparison` directory.
There is no need to move or execute these executables manually.

## Running the Python Scripts

### Notice
Before running the Python scripts, make sure that:
* All the required datasets are placed in the `RankingComparison/provided_data` directory
* The `RankingComparison/generated_data` directory is created for storing the generated intermediate data and indexes of the Python scripts
* The Standard BM25 Search System part is compiled successfully.
* The required Python packages are installed.
* The current working directory is the root `RankingComparison` directory.

### Example Commands

1. **Standard BM25 Search System**:
    ```bash
    # Build the index, generate and save the search results, and evaluate the search results
    $ python3 standard_bm25.py
    ```
    OR
    ```bash
    # Load the saved search results, and evaluate the search results
    # generated: Whether to directly load and evaluate the saved search results or not
    $ python3 standard_bm25.py --generated
    ```

2. **HNSW-based Retrieval System**:
    ```bash
    # Build and save the index, generate the search results, and evaluate the search results
    # M: The number of neighbors for each node during construction
    # efConstruction: The number of neighbors for each node during search
    # efSearch: The number of neighbors for each query during search
    $ python3 ann_hnsw.py --M 8 --efConstruction 200 --efSearch 200
    ```
    OR
    ```bash
    # Load the saved index, generate the search results, and evaluate the search results
    # indexPath: The path to the saved index, filename format: hnsw_index_M_efConstruction_efSearch.faiss
    $  python3 ann_hnsw.py --indexPath generated_data/hnsw_index_8_200_200.faiss
    ```
   
3. **Cascading Rerank System**:
    ```bash
    # Load the saved search results from standard bm25 search system, rerank the search results, and evaluate the reranked search results
    # top: The number of top search results to rerank, in the range of [100, 1000]
    $ python3 cascading_rerank.py --top 100 
    ```

It should be noted that the third script will load the saved search results from the first script.
Therefore, the first script should be run before the third script.

### Known Issues

When running the first `standard_bm25.py` script, the `DataParser`, `IndexBuilder`, and `QueryProcessor` executables will be automatically executed.
In very rare cases, segmentation faults may occur when running these executables.
This is probably because many memory mappings are created and destroyed in a short time in our C code.
If this happens, please just rerun the script.

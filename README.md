# ML Project 2: Preprint Matching with Published Paper

## Team Members
- Harkeerat Singh Sawhney
- Yuchen Chang
- Daniela Gjorgjieva

## Project Description
The ability to identify the published journal for a given preprint is essential for enhancing the accessibility and citation of scientific research. Current methods are often resource-intensive or lack accuracy. Currently, Crossref achieves this mainly by using its own REST API to filter out its candidates and then using Fuzzing similarity to get predict the published Paper of a Preprint. However, this approach has its limitation if a preprint is given which is not part of the Crossref database. This report describes the Dataset creation, and the computational pipeline designed to address this issue by combining fast embeddings, clustering techniques, and LLM-based semantic similarity evaluations. We then highlight and discuss the results which we have achieved.

## Python Libraries install with Conda
```bash
conda env create -f environment.yml
```

## Project Structure
There are two notebooks in the project:
1. `dataset_creation.ipynb`: This notebook contains the code for creating the dataset. It uses the Crossref API to get the published papers for the preprints in the dataset.	
2. `Model Creation`: This notebook contains the code for the computational pipeline. It uses the dataset created in the previous notebook to train the model and evaluate it.
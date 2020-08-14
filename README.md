# Explainable Information Retrieval

## Requirements
 - Python Libraries
```
numpy==1.17.2
torch=1.3.0
transformers==2.3.0
```
 - MARCO-Passage Dataset

download from https://github.com/microsoft/MSMARCO-Passage-Ranking
```
collection.tar.gz
queries.tar.gz
qrels.dev.tsv
qrels.train.tsv
qidpidtriples.train.full.tsv.gz
```
unzip all of them into data/marco_passage/

## Train
```python -m main```

## Visualize
```python -m visual```


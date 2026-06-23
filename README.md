# NERforRG

Supporting code and data for the manuscript **"A natural language processing framework for 
mining paleoclimate literature and its application for global synthesis 
of 2000-year climate change reconstructions"**.

The repository turns the final, geolocated climate-anomaly entities into
annual voting counts, the Temperature Voting Index (TVI), the Precipitation
Voting Index (PVI) and spatial-sensitivity results.

## Repository layout

```text
data/
  annotations/       Annotated corpus
  reference/    External reconstruction series used for comparison
  search_keywords.xlsx/	List of keywords used by the crawler program
src/            Reusable command-line analysis scripts
configs/        Paper-compatible NER experiment configuration
results/        Reproduced tables, annual series
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_entities.py
python src/compute_indices.py
python src/sensitivity_analysis.py
python src/plot_indices.py
```

## NER model training

The repository now includes a PyTorch reconstruction of the three architectures
compared in manuscript Table 3: BERT-base, BERT-CRF, and BERT-BiLSTM-CRF. They
share one document split, tokenizer, alignment pipeline, and exact-span metric
implementation. Install the separate NER environment and run:



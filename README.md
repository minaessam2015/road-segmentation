# Road Segmentation

Initial scaffold for the Hudhud take-home project.

## Layout

- `notebooks/`: exploratory analysis and experiments
- `data/`: local dataset storage
- `src/road_segmentation/`: reusable project code
- `scripts/`: thin command-line entrypoints
- `tests/`: initial automated tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

```bash
python scripts/download_data.py
```

The dataset will be downloaded into `data/raw/deepglobe-road-extraction-dataset/`.

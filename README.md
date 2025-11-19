# Census Income Classification & Segmentation

## 1. Objective

Using weighted census microdata, this project:

1. Trains and evaluates a classifier to predict whether an individual earns more than \$50,000.
2. Builds a segmentation model (clustering) to group individuals into distinct marketing segments.

## 2. Project Structure

- `data/`
  - `censusbureau.data` – raw data (comma-separated)
  - `census-bureau.columns` – one column name per line
- `src/`
  - `data_utils.py` – common data loading and preprocessing utilities
  - `train_classifier.py` – code to train and evaluate the income classifier
  - `segmentation.py` – code to build and inspect the segmentation model
- `classifier.joblib` – saved trained classifier (created after training)

## 3. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

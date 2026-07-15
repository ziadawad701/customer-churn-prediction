# Customer Churn Prediction — Mini ML Pipeline

A small end-to-end ML pipeline: data collection → EDA → preprocessing →
model training → evaluation → Streamlit deployment.

## Project structure
```
churn_project/
├── data/
│   └── telco_churn.csv          # synthetic Telco-style dataset (3000 rows, 14 cols)
├── model/
│   ├── churn_model.pkl          # trained classifier model
│   ├── encoders.pkl             # LabelEncoders for categorical columns
│   ├── feature_columns.pkl      # column order used at training time
│   └── model_comparison.csv     # model comparison metrics
├── plots/                       # EDA + evaluation charts
├── pipeline.py                  # full pipeline: import -> EDA -> preprocess -> train -> evaluate
├── app.py                       # Streamlit deployment app
├── requirements.txt
├── .gitignore
└── README.md
```

## Results
See `model/model_comparison.csv` for full metrics across the models tried
(accuracy, precision, recall, F1, ROC-AUC). The best-performing model is
saved as `model/churn_model.pkl` and used by the Streamlit app.

## Setup
```bash
pip install -r requirements.txt
```

## Run the pipeline (regenerate data, plots, and model)
```bash
python pipeline.py
```

## Run the Streamlit app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Notes
- The dataset is synthetically generated to mirror the well-known Kaggle
  "Telco Customer Churn" schema (same columns/types). Swap in the real CSV
  any time — `pipeline.py` only assumes matching column names.

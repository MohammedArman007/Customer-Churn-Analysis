# train_model.py
"""
Train the RandomForest model on your CSV and save the pipeline.
Usage:
    python train_model.py --data data/customer_churn_dataset.csv
"""

import argparse
import os
from utils import load_data, train_save_model

def main(data_path, model_path='model.joblib', pipeline_path='pipeline.joblib'):
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    print("Training model (this may take some time)...")
    metrics = train_save_model(df, model_path=model_path, pipeline_path=pipeline_path)
    print("Training finished.")
    print("Train accuracy: {:.4f}".format(metrics['train_score']))
    print("Test accuracy:  {:.4f}".format(metrics['test_score']))
    print(f"Saved pipeline to {model_path} and preprocessor to {pipeline_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV data")
    parser.add_argument("--model", default="model.joblib", help="Output model path")
    parser.add_argument("--pipeline", default="pipeline.joblib", help="Output preprocessor path")
    args = parser.parse_args()
    main(args.data, model_path=args.model, pipeline_path=args.pipeline)

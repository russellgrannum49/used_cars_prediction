# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""
import os
import argparse
import logging
import mlflow.sklearn
import pandas as pd
import sys
from pathlib import Path

mlflow.start_run()  # Starting the MLflow experiment run

def main():
    # Argument parser setup for command line arguments
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")  # Path to the trained model artifact
    args = parser.parse_args()
    print("input model path:", args.model)
    print ("files in input path:", os.listdir(args.model))

    # Load the trained model from the provided path
    model_path= args.model
    if "model" in os.listdir(args.model):
        model_path = os.path.join(args.model, "model")
    model = mlflow.sklearn.load_model(args.model_path)  # code to load model from args.model)

    print("Registering the best trained used cars price prediction model")

    # Register the model in the MLflow Model Registry under the name "price_prediction_model"
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="price_prediction_model",  # name under which the model will be registered
        artifact_path="random_forest_price_regressor"  # path where the model artifacts will be stored
    )

    # End the MLflow run
    mlflow.end_run()  # code to end the MLflow run
    

if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Path to teh trained model ')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

 # Step 1: Load the model from the specified path
   
    model = mlflow.sklearn.load_model(args.model_path)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

 # Step 2: Log the loaded model in MLflow
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)

# Step 3: Register the logged model and retrieve its version
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)

 # Step 4: Write model registration details into a JSON file 
    model_info = {
        "model_name": args.model_name,
        "model_version": mv.version
    }   
# Ensure the output directory exists
    os.makedirs(os.path.dirname(args.model_info_output_path), exist_ok=True)
    
    with open(args.model_info_output_path, "w") as f:
        json.dump(model_info, f)
    
    print(f"Model {args.model_name} version {mv.version} registered successfully.")
    # -----------  WRITE YOR CODE HERE -----------
    
    # Step 1: Load the model from the specified path using `mlflow.sklearn.load_model` for further processing.  
    # Step 2: Log the loaded model in MLflow with the specified model name for versioning and tracking.  
    # Step 3: Register the logged model using its URI and model name, and retrieve its registered version.  
    # Step 4: Write model registration details, including model name and version, into a JSON file in the specified output path.  


if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
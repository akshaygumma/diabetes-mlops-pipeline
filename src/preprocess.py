import pandas as pd
import sys
import os
import yaml

##Loading the parameters from the params.yaml file
params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    input_path = params['input']
    output_path = params['output']
    preprocess_data(input_path, output_path)



##Note : This script reads a CSV file, removes the header, and saves the processed data to a new CSV file. The input and output file paths are specified in a params.yaml file.
## There is no feature engineering or data cleaning in this basic example, but you can expand it as needed.


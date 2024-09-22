import os
import argparse
from utils.JobDataset import JobDataset
from utils.model import create_model

MODEL_NAME = 'roberta-base'
NUM_LABELS = 6
TOKENIZER_LENGTH = 128

def predict_job_titles(titles, weights_path):
    model = create_model(MODEL_NAME, NUM_LABELS, TOKENIZER_LENGTH)
    model.load_weights(weights_path)

    dataset_predict = JobDataset()
    for title in titles:
        x = dataset_predict.preprocess_title(title)
        x = dataset_predict.tokenize(x)
        x = model.predict(x)[0]
        print(x)
        print(f"{title}: {dataset_predict.predict_labels(x)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict job titles")
    parser.add_argument("-t", "--titles", nargs="+", required=True, help="Job titles to predict (separated by spaces)")
    parser.add_argument("-w", "--weights", default="./weights/weights_f1_07.h5", help="Path to the weights file")
    args = parser.parse_args()

    weights_path = os.path.abspath(args.weights)

    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        exit(1)

    predict_job_titles(args.titles, weights_path)
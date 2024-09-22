# Job Title Classification Project

This project focuses on classifying job titles into multiple categories using a deep learning approach. Below, key aspects of the project are outlined.

> **Note:** Initial Approach (Before Deep Learning)
> 
> Before settling on the deep learning approach, I have initially explored an ensemble model consisting of 6 binary classifiers:
> 
> 1. Used Lazy Predict to identify the most promising model for each label.
> 2. Hyperparameter search was conducted to find the optimal parameters for each classifier.
> 3. While some classifiers performed well, others yielded poor results.
> 
> This initial approach provided valuable insights:
> - The varying performance across different labels highlighted the complexity of the task.
> - It suggested that a more sophisticated model might be necessary for consistent performance across all categories.

> **Note:** Current Approach
>
> The current approach utilizes transfer learning with a pre-trained transformer model (RoBERTa), fine-tuned for the job title classification task. More details on this approach are provided in the following sections.

> **Note:** Project Notebooks
>
> - `notebooks/roberta.ipynb`: This notebook contains the code for fine-tuning the RoBERTa transformer model for job title classification.
> - `notebooks/classifiers.ipynb`: This notebook includes the code for the ensemble of classifiers approach, along with minimal data analysis (single graph :) ).


## 0. How to run the code?

### Python Version
This project was developed using Python 3.10.15. It's recommended to use this version or later to ensure compatibility.

### Setting Up a Virtual Environment
To set up and run the code, follow these steps:

1. Create a virtual environment:
   ```
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Code
Before running the code, place the JobLevelData.xlsx file in the root directory of the project. After setting up the virtual environment and installing the dependencies, you can run the main script:

   ```
   python infer.py -t "Job Title 1" "Job Title 2" ... -w path/to/weights.h5
   ```

   - The `-t` or `--titles` argument is required and should be followed by one or more job titles you want to classify. Enclose each title in quotes if it contains spaces.
   - The `-w` or `--weights` argument is optional and specifies the path to the model weights file. If not provided, it defaults to `./weights/weights_f1_07.h5`.

To run the training script, use the following command:

   ```
   python train.py
   ```
The best model and weights will be saved to the `./weights` and './models' directory. Tensorboard logs will be saved to the `./logs` directory.

## 1. Data Preprocessing

### Missing Values and Class Imbalance
- I have filled in some missing values based on the job title and internet research.
- Added some extra titles for the bussiness Owner category because this label was severely underrepresented.

The data preprocessing pipeline involves several steps to clean and prepare the dataset for optimal model performance:

### Text Cleaning
- Lowercase conversion
- Special character and digits removal
- Extra whitespace elimination

### Text Normalization
- Stop word removal (excluding 'it')
- Job title normalization (e.g., correcting misspellings, expanding abbreviations)

### Tokenization
I have used the RoBERTa tokenizer for processing job titles:
- Tokenization with a maximum length of 128 tokens
- Padding/truncation to ensure uniform input size
- Generation of attention masks

### Target Processing
- The target columns were one-hot encoded.

## 2. Model and Architecture Selection

I have chosen a Transformer-based model, specifically RoBERTa, for this classification task due to its:
- Superior performance in NLP tasks
- Ability to capture complex contextual information
- Suitability for multi-label classification

The model architecture consists of:
- A pre-trained RoBERTa base model
- A custom classification head with sigmoid activation for multi-label output
- Input layers for both token IDs and attention masks

## 3. Training and Testing the Model

### Dataset Splitting
I have used an 80-20 split for training and validation sets, however, due to the class imbalance in the dataset, it is worth implementing some stratification techniques to make sure that the each label is equally represented in both sets.

### Performance Metrics
During the evaluation process, Accuracy, AUC, Precision, Recall and F1 were used. However, since false positives and false negatives are equally important for this task, the F1 score was chosen as the primary metric.

## 4. Interpreting Results

The model's output:
- Each output neuron corresponds to a single job title category with value from 0 to 1
- A threshold of 0.5 is used to determine if a category is assigned

## 5. Documenting Desicions
- Early stopping to prevent overfitting
- Learning rate reduction on plateau to fine-tune training
- Model checkpointing to save the best performing model
- TensorBoard integration for performance visualization

## 6. Results and Conclusions

### Model Output Quality
The model achieved an overall F1 score of 0.7, which indicates reasonably good performance but leaves room for improvement. This score suggests that the model has a balanced ability to correctly identify positive cases (precision) and find all positive cases (recall) across the different job title categories.

### Strengths and Weaknesses
Strengths:
- When checking the results in the validation set, the model's performance was satisfactory.
- Fast and efficient training process (around 15-20 minutes on a P100 GPU)

Weaknesses:
- The current F1 score of 0.7 suggests that there's still a significant margin for error in classifications.
- The model may struggle with underrepresented categories due to class imbalance.

### Recommendations for Improvement

1. Label Stratification: To address the class imbalance issue, it is crucial to implement label stratification when splitting the dataset. This ensures that each label is equally represented in both the training and validation sets, which should significantly improve the model's performance across all categories.

2. Expand the Dataset: A larger dataset would likely benefit the model's quality. More diverse examples for each category can help the model learn more robust representations and improve its generalization capabilities.

3. Incorporate Additional Data: If the data was scraped from LinkedIn, it is worith considering to collect job descriptions or profile descriptions in the analysis. This additional context could provide valuable information for more accurate classification.

4. Ensemble Methods: Consider creating an ensemble of models, possibly combining the current deep learning approach with traditional machine learning models for specific categories where they performed well.



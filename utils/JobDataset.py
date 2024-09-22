import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer

class JobDataset:
    def __init__(self, file_path='./JobLevelData.xlsx', model_name='roberta-base', tokenizer_length=128, batch_size=32, add_extra_data=True, preprocess_nans=True):
        self.file_path = file_path
        self.model_name = model_name
        self.tokenizer_length = tokenizer_length
        self.batch_size = batch_size
        self.add_extra_data = add_extra_data
        self.preprocess_nans = preprocess_nans
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels_num = None
        self.label_mapping = {
            0: 'Vice President',
            1: 'Manager',
            2: 'Individual Contributor/Staff',
            3: 'Chief Officer',
            4: 'Owner',
            5: 'Director'}

    def load_data(self):
        self.df = pd.read_excel(self.file_path)
        if self.preprocess_nans:
            self._handle_nans()
        if self.add_extra_data:
            self._add_extra_data()
        self._preprocess_data()
        self._one_hot_encode()


    def _handle_nans(self):
        self.df.loc[29, 'Column 1'] = 'Chief Officer'                        # CINO
        self.df.loc[829, 'Column 1'] = 'Individual Contributor/Staff'        # Release of Information Tech II
        self.df.loc[1406, 'Column 1'] = 'Manager'                            # Global People Systems, Processes and Information Manager
        self.df.loc[1713, 'Column 1'] = 'Individual Contributor/Staff'       # Supplier Quality Enginee
        self.df.loc[1785, 'Column 1'] = 'Manager'                            # RC Environmental and Cyber Specialized Subscription Manager
        self.df.loc[2182, 'Column 1'] = 'Director'                           # Senior IndependeDirector and Chair of the Customer and Communities Network
        self.df.loc[2182, 'Title'] = 'Senior Independent Director and Chair of the Customer and Communities Network'        # Senior IndependeDirector and Chair of the Customer and Communities Network

    def _add_extra_data(self):
        new_rows = [
            {'Title': 'CEO & CO-OWNER', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'CEO & Founder', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'CEO & CO-founder', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'Founder, CTO', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'Founder', 'Column 1': 'Owner'},
            {'Title': 'Founder & CEO', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'Founder and visionar', 'Column 1': 'Owner'},
            {'Title': 'CEO/Founder', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'CEO & founder', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
            {'Title': 'Chief Business Development Officer & Co-Founder', 'Column 1': 'Owner', 'Column 2': 'Chief Officer'},
        ]
        self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

    def _preprocess_data(self):
        self.df['Title'] = self.df['Title'].apply(self.preprocess_title)


    def preprocess_title(self, title):
        # Lowercase and remove special characters
        title = title.lower()
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s{2,}', ' ', title)

        # Remove stop words
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        stop_words.remove('it')
        title = ' '.join([word for word in title.split() if word not in stop_words])
        
        # Normalize job titles
        job_title_dict = {
            # misspelings
            'vise': 'vice',
            'senoir': 'senior',
            'maneger': 'manager',
            'assistent': 'assistant',
            'supervisr': 'supervisor',
            'analist': 'analyst',
            'enginere': 'engineer',
            'developr': 'developer',
            'programer': 'programmer',
            'acountant': 'accountant',
            'lawer': 'lawyer',
            'docter': 'doctor',
            'analista': 'analyst',
            # short forms
            'dev': 'developer',
            'dev.': 'developer',
            'eng': 'engineer',
            'eng.': 'engineer',
            'engr': 'engineer',
            'engr.': 'engineer',
            # title prefixes 
            'reg': 'regional',
            'reg.': 'regional',
            'assoc': 'associate',
            'assoc.': 'associate',
            'asst': 'assistant',
            'asst.': 'assistant',
            'exec': 'executive',
            'exec.': 'executive',
            'deputy': 'deputy',
            'deputy.': 'deputy',
            'mng': 'managing',
            'mng.': 'managing',
            # expertise level
            'sr': 'senior',
            'sr.': 'senior',
            'snr': 'senior',
            'snr.': 'senior',
            'sen': 'senior',
            'sen.': 'senior',
            'jr': 'junior',
            'jr.': 'junior',
            'jnr': 'junior',
            'jnr.': 'junior',
            'jun': 'junior',
            'jun.': 'junior',
            'mid': 'middle',
            'mid.': 'middle',
            'mdl': 'middle',
            'mdl.': 'middle',
            # vice president
            'vp': 'vice president',
            'svp': 'senior vice president',
            'evp': 'executive vice president',
            'avp': 'assistant vice president',
            'sevp': 'senior executive vice president',
            'gvp': 'group vice president',
            'dvp': 'divisional vice president',
            'rvp': 'regional vice president',
            'cvp': 'corporate vice president',
            'davp': 'deputy assistant vice president',
            'savp': 'senior assistant vice president',
            'mvp': 'managing vice president',
            'arvp': 'associate regional vice president',
            # c level officers
            'ceo': 'chief executive officer',
            'cfo': 'chief financial officer',
            'coo': 'chief operating officer',
            'cto': 'chief technology officer',
            'cio': 'chief information officer',
            'chro': 'chief human resources officer',
            'cdo': 'chief data officer',
            'cmo': 'chief marketing officer',
            'cso': 'chief sales officer',
            'cco': 'chief communications officer',
            'cro': 'chief relationship officer',
            # directors
            'dir': 'director',
            'dir.': 'director',
            # managers
            'mgr': 'manager',
            'mgr.': 'manager',
            'mng': 'manager',
            'mng.': 'manager',
            'mngr': 'manager',
            'mngr.': 'manager',
        }
        
        title = ' '.join([job_title_dict.get(word, word) for word in title.split()])
        return title

    def _one_hot_encode(self):
        columns_to_encode = ['Column 1', 'Column 2', 'Column 3', 'Column 4']
        one_hot_encoded = pd.DataFrame()
        unique_values = set()
        for column in columns_to_encode:
            values = self.df[column].dropna().unique().tolist()
            unique_values.update(values)
        self.labels_num = len(unique_values)
        self.label_mapping = {}  # Add this line to store the label mapping
        for i, value in enumerate(unique_values):
            one_hot_encoded[f'Label_{value}'] = self.df[columns_to_encode].eq(value).any(axis=1).astype(int)
            self.label_mapping[i] = value  # Add this line to create the mapping
        self.df = pd.concat([self.df['Title'], one_hot_encoded], axis=1)

    def tokenize(self, text):
        input_ids = []
        attention_masks = []
        tokens = self.tokenizer.encode_plus(text, max_length=self.tokenizer_length,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')   
         
        input_ids.append(np.asarray(tokens["input_ids"]).reshape(self.tokenizer_length,))
        attention_masks.append(np.asarray(tokens["attention_mask"]).reshape(self.tokenizer_length,))

        return (np.asarray(input_ids), np.asarray(attention_masks))

    
    def get_train_validation_split(self, test_size=0.2, random_state=42):
        X = self.df['Title'].values
        y = self.df.drop('Title', axis=1).values
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Tokenize the text data
        X_train_tokenized = []
        X_valid_tokenized = []
        
        for text in X_train:
            X_train_tokenized.append(self.tokenize(text))

        for text in X_valid:
            X_valid_tokenized.append(self.tokenize(text))

        # Separate input_ids and attention_masks
        X_train_input_ids = np.array([item[0][0] for item in X_train_tokenized])
        X_train_attention_masks = np.array([item[1][0] for item in X_train_tokenized])
        X_valid_input_ids = np.array([item[0][0] for item in X_valid_tokenized])
        X_valid_attention_masks = np.array([item[1][0] for item in X_valid_tokenized])

        return (
            X_train_input_ids,
            X_train_attention_masks,
            X_valid_input_ids,
            X_valid_attention_masks,
            y_train.astype(np.float32),
            y_valid.astype(np.float32)
        )

    def predict_labels(self, prediction_array, threshold=0.5):
        if len(prediction_array) != self.labels_num:
            raise ValueError(f"Prediction array length ({len(prediction_array)}) does not match the number of labels ({self.labels_num})")
        
        predicted_labels = []
        for i, prob in enumerate(prediction_array):
            if prob >= threshold:
                predicted_labels.append(self.label_mapping[i])
        
        return predicted_labels

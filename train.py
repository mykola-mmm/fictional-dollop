import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from utils.JobDataset import JobDataset
from utils.model import create_model
from utils.utils import plot_training_history

MODEL_NAME = 'roberta-base'
NUM_LABELS = 6
EPOCH = 2
PATIENCE_CALLBACK = 20
LR_CALLBACK = 10
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
TOKENIZER_LENGTH = 128

# Load and preprocess data
dataset = JobDataset('./JobLevelData.xlsx')
dataset.load_data()
train_input_ids, train_attention_masks, valid_input_ids, valid_attention_masks,y_train,y_valid = dataset.get_train_validation_split(test_size=0.2, random_state=42)


os.makedirs('./models', exist_ok=True)
os.makedirs('./weights', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Using OneDeviceStrategy with GPU.")
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
else:
    print("No GPU available. Using default strategy for CPU computing.")
    strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = create_model(MODEL_NAME, NUM_LABELS, TOKENIZER_LENGTH)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    acc = tf.keras.metrics.BinaryAccuracy('accuracy')
    auc = tf.keras.metrics.AUC(name='auc')
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    f1_score = tf.keras.metrics.F1Score(average='macro', name='f1_score')

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[acc, auc, precision, recall, f1_score]
    )
    model.summary()

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './models/best_model.keras',
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Checkpoint for saving only the weights
    weights_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './weights/best_model_weights.h5',
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score',
        mode='max',
        patience=PATIENCE_CALLBACK,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score',
        factor=0.5,
        patience=LR_CALLBACK,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    
    history = model.fit(
        (train_input_ids, train_attention_masks),
        y_train,
        validation_data=((valid_input_ids,valid_attention_masks), y_valid),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=[model_checkpoint, weights_checkpoint, tensorboard_callback, early_stopping, reduce_lr]
    )
    plot_training_history(history)



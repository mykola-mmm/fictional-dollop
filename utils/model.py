import tensorflow as tf
from transformers import TFRobertaForSequenceClassification

def create_model(model_name, num_labels, tokenizer_length):
    transformer_model = TFRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    input_ids = tf.keras.layers.Input(shape=(tokenizer_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(tokenizer_length,), dtype=tf.int32, name='attention_mask')

    embeddings = transformer_model(input_ids, attention_mask=attention_mask)[0]
    output = tf.keras.layers.Activation('sigmoid')(embeddings)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return model
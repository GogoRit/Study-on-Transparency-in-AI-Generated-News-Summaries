"""DSCI-601_BiasDetection.ipynb

Installing required dependencies
"""

#pip install transformers

#pip install datasets

#pip install detoxify

"""Importing files"""

from datasets import load_dataset, concatenate_datasets
import pandas as pd
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification,RobertaTokenizer,AutoTokenizer, TFAutoModelForSequenceClassification,pipeline,TextClassificationPipeline,AutoModelForSequenceClassification,BertTokenizer, TFBertForSequenceClassification
from prettytable import PrettyTable
from detoxify import Detoxify

"""Using dataset *cnn_dailymail* for running bias models"""

combined_dataset = concatenate_datasets([load_dataset('cnn_dailymail', '1.0.0')['train']])

print("Combined Length")
print(len(combined_dataset))

"""Put all highlights(summaries of new articles) in a dataframe"""

df_train_highlights = pd.DataFrame({'highlights': [article['highlights'] for article in combined_dataset]})

"""Using 1st value to test models"""

predict_val=df_train_highlights['highlights'].iloc[0]

# BIAS https://huggingface.co/d4data/bias-detection-model

tokenizer_bias = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model_bias = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

classifier_bias = pipeline('text-classification', model=model_bias, tokenizer=tokenizer_bias) # cuda = 0,1 based on gpu availability

# TOXIC https://huggingface.co/unitary/toxic-bert
tokenizer_toxic1= AutoTokenizer.from_pretrained("unitary/toxic-bert")
model_toxic1 = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
classifier__toxic1 = pipeline('text-classification', model=model_toxic1, tokenizer=tokenizer_toxic1) # cuda = 0,1 based on gpu availability

# TOXIC https://huggingface.co/martin-ha/toxic-comment-model
tokenizer_toxic = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
model_toxic = AutoModelForSequenceClassification.from_pretrained("martin-ha/toxic-comment-model")

classifier__toxic =  TextClassificationPipeline(model=model_toxic, tokenizer=tokenizer_toxic)

# Load pre-trained BERT model and tokenizer
tokenizer_bert_base_uncased = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert_base_uncased = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load pre-trained RoBERTa model and tokenizer
tokenizer_roberta_base = RobertaTokenizer.from_pretrained("roberta-base")
model_roberta_base = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=9)

# Load pre-trained RoBERTa model and tokenizer
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = predict_val
max_score_label = max(classifier(sentences)[0], key=lambda x: x['score'])

# Tokenize the input text for both models
inputs_bert = tokenizer(predict_val, padding=True, truncation=True, max_length=128, return_tensors="tf")
inputs_roberta = tokenizer(predict_val, padding=True, truncation=True, max_length=128, return_tensors="tf")

# Perform inference for both models
outputs_bert = model_bert_base_uncased(inputs_bert)
outputs_roberta = model_roberta_base(inputs_roberta)

# Get predicted labels for both models
predictions_bert = tf.nn.softmax(outputs_bert.logits, axis=-1)
predictions_roberta = tf.nn.softmax(outputs_roberta.logits, axis=-1)

# Get predicted label for BERT-based model
label_bert = "toxic" if tf.argmax(predictions_bert, axis=1).numpy()[0] == 1 else "not toxic"

# Get predicted label for RoBERTa-based model
label_roberta = "toxic" if tf.argmax(predictions_roberta, axis=1).numpy()[0] == 1 else "not toxic"

# Perform inference and get predicted emotion category for BERT-based model
outputs_bert = model_bert_base_uncased(inputs)
predicted_label_id_bert = tf.argmax(outputs_bert.logits, axis=1).numpy()[0]
predicted_label_bert = emotion_labels[predicted_label_id_bert]

# Perform inference and get predicted emotion category for RoBERTa-based model
outputs_roberta = model_roberta_base(inputs)
predicted_label_id_roberta = tf.argmax(outputs_roberta.logits, axis=1).numpy()[0]
predicted_label_roberta = emotion_labels[predicted_label_id_roberta]

# Tokenize the input text
inputs = tokenizer(predict_val, padding=True, truncation=True, max_length=128, return_tensors="tf")

# Perform inference and get predicted label for BERT-based model
outputs_bert = model_bert_base_uncased(inputs)
predictions_bert = tf.nn.softmax(outputs_bert.logits, axis=-1)
label_bert = "sexist" if tf.argmax(predictions_bert, axis=1).numpy()[0] == 1 else "not sexist"

# Perform inference and get predicted label for RoBERTa-based model
outputs_roberta = model_roberta_base(inputs)
predictions_roberta = tf.nn.softmax(outputs_roberta.logits, axis=-1)
label_roberta = "sexist" if tf.argmax(predictions_roberta, axis=1).numpy()[0] == 1 else "not sexist"

"""Outputs of each model"""

print(f"Summary String: {predict_val}")

from prettytable import PrettyTable

# Create a PrettyTable instance
table = PrettyTable()

# Add columns to the table
table.field_names = ["Category", "Response"]

# Add data to the table
table.add_row(["Bias Detection Model", classifier_bias(str(predict_val))])
table.add_row(["Max Value Response from Roberta", max_score_label])
table.add_row(["Toxic BERT", classifier__toxic1(str(predict_val))])
table.add_row(["Toxic Comment Model", classifier__toxic(str(predict_val))])
table.add_row(["Response from BERT-based model", label_bert])
table.add_row(["Response from RoBERTa-based model", label_roberta])
table.add_row(["Predicted Emotion Category from BERT", predicted_label_bert])
table.add_row(["Predicted Emotion Category from RoBERTa", predicted_label_roberta])
table.add_row(["Response from BERT-based model", label_bert])
table.add_row(["Response from RoBERTa-based model", label_roberta])

# Print the table
print(table)

# Initialize Detoxify models
original_model = Detoxify('original')
unbiased_model = Detoxify('unbiased')
multilingual_model = Detoxify('multilingual')

# Get predictions from all three models
original_results = original_model.predict(predict_val)
unbiased_results = unbiased_model.predict(predict_val)
multilingual_results = multilingual_model.predict(predict_val)

# Create a PrettyTable instance
table = PrettyTable()

# Add columns to the table
table.field_names = ["Model", "Category", "Probability"]

# Add data to the table
for model_name, results in [("Original", original_results.items()), ("Unbiased", unbiased_results.items()), ("Multilingual", multilingual_results.items())]:
    for category, probability in results:
        table.add_row([model_name, category, probability])

# Print the table
print(table)
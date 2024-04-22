# Capstone_Project

This repository contains code for preprocessing textual data and performing bias detection using various NLP models.

## Table of Contents

- [Preprocessing](#preprocessing)
- [Summarization and Model Training](#summarization-and-model-training)
- [Bias Detection](#bias-detection)
- [Requirements](#requirements)

## Preprocessing

The `dsci_601_preprocessing.py` script provides a pipeline for preprocessing textual data, including tasks such as text cleaning, tokenization, stop words removal, and more. The preprocessing pipeline includes the following steps:

1. **Text Cleaning**: Removal of URLs, special characters, and unwanted patterns using regex.
2. **Contractions Expansion**: Expansion of contractions (e.g., "don't" to "do not").
3. **Tokenization and Stop Words Removal**: Tokenization of text into words and removal of stop words using NLTK.
4. **Short Sentences Removal**: Removal of short sentences based on a minimum length threshold.
5. **Tags Removal**: Removal of specified tags from the text.

The pipeline can be applied to any textual dataset to prepare it for further analysis or modeling.

## Summarization and Model Training

The `dsci_601_summarizationpipeline.py` script demonstrates the process of summarization and model training using various NLP models. It includes the following steps:

1. **Important Words Extraction**: Extraction of important words from articles using TF-IDF.
2. **Custom Prompt Generation**: Generation of custom prompts for summarization using the extracted important words.
3. **Summarization with LLAMA-2**: Summarization of articles using the LLAMA-2 model.
4. **Zero-Shot Prompting**: Summarization of conversations using zero-shot prompting with pre-trained language models.
5. **Future Work**: Exploration of additional ways to improve summarization techniques, including:
    - **Better Important Words Detection**: Investigate methods for more accurate detection of important words for summarization.
    - **Enhanced Computing Resources**: Utilize better computing resources for faster and more efficient model training and inference.
    - **Experimentation with Different Approaches**: Try different approaches and models for text summarization to achieve better results.

The script showcases how to leverage pre-trained models for text summarization tasks and generate human-readable summaries.

## Bias Detection

The `dsci_601_biasdetection.py` script focuses on detecting biases in textual data using various NLP models. It includes the following steps:

1. **Data Loading**: Loading of textual data from the CNN/Daily Mail dataset.
2. **Bias Detection Models**: Utilization of pre-trained models for bias detection, including a bias detection model, toxic comment classifiers, emotion classifiers, and Detoxify models.
3. **Political Bias Detection**: Ongoing work on detecting political bias in textual data.
4. **Model Tuning**: Tuning of bias detection models as per specific requirements.
5. **Future Work**: Exploration of additional ways to study bias detection, including adding political bias detection, tuning models according to specific requirements, and trying different approaches for improvement.

### Requirements

- Python version: 3.9.7
- Accelerate==0.6.0
- BitsAndBytes==0.0.4
- Datasets==1.16.0
- Detoxify==0.2.2
- NLTK==3.6.5
- Pandas==1.4.0
- PrettyTable==2.2.0
- Scikit-learn==0.24.2
- TensorFlow==2.7.0
- Torch==1.10.0
- Transformers==4.17.0

The dataset used is the CNN dataset from the Hugging Face library. [Link to Dataset](https://huggingface.co/datasets/cnn_dailymail)

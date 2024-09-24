## Project Structure

The repository is structured as follows:

```plaintext
.
├── data/               # Raw and processed data
│   ├── raw/            # Original dataset
│   └── processed/      # Processed data
├── notebooks/          # Jupyter notebooks
├── src/                # Source code for models, evaluation, and preprocessing
│   ├── models/         # Model-related scripts (e.g., GPT-4, LLaMA)
│   ├── evaluation/     # Bias and hallucination detection scripts
│   ├── preprocessing/  # Data preprocessing scripts
│   └── utils/          # Helper functions
├── results/            # Generated summaries and evaluation metrics
│   ├── summaries/      # LLM-generated summaries
│   ├── metrics/        # Evaluation metrics (e.g., bias, hallucination)
│   └── figures/        # Visualizations
├── experiments/        # Experiment logs and results
│   ├── experiment_1_gpt4_abortion_rights/
│   └── experiment_2_llama_abortion_rights/
├── docs/               # Project documentation
└── tests/              # Unit tests


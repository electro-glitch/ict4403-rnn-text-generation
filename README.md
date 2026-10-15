# GPU-Optimized Recurrent Language Models on the Nietzsche Corpus

**Student:** Tanay Aggarwal  
**Registration number:** 230911278  
**Module:** ICT4403 - Foundations of Generative AI, Assignment 2

This project compares Simple RNN, Bidirectional RNN, LSTM, and GRU models for word-level autoregressive generation under controlled conditions. The complete implementation, results, plots, generated samples, and critical discussion are contained in `ICT4403_Assignment2_Autoregressive_Text_Generation.ipynb`.

## Dataset

The project uses the **Nietzsche philosophical-text corpus** distributed for Keras's long-running neural text-generation example:

- Official example: https://keras.io/examples/generative/lstm_character_level_text_generation/
- Direct corpus: https://s3.amazonaws.com/text-datasets/nietzsche.txt

The corpus is deliberately less conventional than news, Wikipedia, or children's fiction. Its aphorisms, long argumentative sentences, rhetorical questions, capitalized emphasis, and abstract vocabulary make coherence and long-context modelling difficult. It is still a recognized text-generation comparison corpus: the official Keras example has used it since 2015.

## GPU implementation

The verified environment uses Python 3.12.7, PyTorch 2.13.0+cu126, CUDA 12.6, cuDNN, and the NVIDIA GeForce RTX 3050 Laptop GPU. The models use PyTorch's native `nn.RNN`, bidirectional `nn.RNN`, `nn.LSTM`, and `nn.GRU` modules, which dispatch to optimized cuDNN kernels on CUDA. A real forward/backward optimizer smoke test and the recorded backend are included in the notebook.

Performance-oriented choices include cuDNN recurrent kernels, automatic mixed precision, fused AdamW, large batches, pinned/non-blocking transfers, label-smoothed training, gradient clipping, learning-rate reduction, early stopping, and checkpoint restoration. The same settings are used for all models.

Each architecture is fine-tuned with an equal four-candidate validation-only search over hidden-state size, dropout, learning rate, and weight decay. The test split remains untouched during selection. Winning configurations are reinitialized, trained for the full budget, and evaluated once on the test set.

## Run

```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook ICT4403_Assignment2_Autoregressive_Text_Generation.ipynb
```

Select the **Python 3.12 (FGAI CUDA)** kernel and run **Kernel -> Restart & Run All**. The corpus is already included under `data/`, so the notebook runs offline. CUDA is selected automatically; the notebook falls back to CPU only when CUDA is unavailable.

## Outputs

The executed notebook writes:

- `artifacts/test_metrics.csv`
- `artifacts/tuning_trials.csv`
- `artifacts/tuned_hyperparameters.csv`
- `artifacts/tuning_impact.csv`
- `artifacts/generations.csv`
- `artifacts/generation_quality.csv`
- `artifacts/experiment_summary.json`
- `artifacts/training_curves.png`
- `artifacts/tuning_validation.png`
- `artifacts/model_efficiency_diagnostics.png`
- `artifacts/pareto_tradeoffs.png`
- `artifacts/additional_diagnostics.csv`
- `artifacts/test_comparison.png`
- `artifacts/temperature_analysis.png`
- model checkpoints under `checkpoints/`

Test perplexity is the primary model-selection metric. Accuracy, top-5 accuracy, training time, parameter count, diversity, repetition, prompt overlap, and qualitative reading are treated as supporting evidence.


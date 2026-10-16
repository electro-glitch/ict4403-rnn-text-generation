# GPU-Optimized Word-Level Autoregressive Text Generation

**Student:** Tanay Aggarwal  
**Registration number:** 230911278  
**Module:** ICT4403 - Foundations of Generative AI, Assignment 2

This repository contains the complete implementation, executed results, plots, generated samples, and analysis for the assignment. The main deliverable is the Jupyter Notebook `ICT4403_Assignment2_Autoregressive_Text_Generation.ipynb`.

## Problem statement and objective

The objective is to build and compare four recurrent neural-network architectures for word-level autoregressive text generation: Simple RNN, Bidirectional RNN (BiRNN), LSTM, and GRU. For one prepared textual dataset, the work must construct input-target sequences, tune and train every model, evaluate them quantitatively and qualitatively, and study generation from identical seeds at multiple sampling temperatures. The final goal is to identify the model that offers the most effective balance of predictive performance, computational efficiency, coherence, diversity, contextual consistency, and low repetition.

## Dataset description and source

The selected dataset is the **Nietzsche philosophical-text corpus**. It is atypical compared with commonly used news, Wikipedia, or children's-fiction corpora, but it remains useful for comparing language models because it contains difficult abstract vocabulary, aphorisms, rhetorical questions, long argumentative sentences, and varied punctuation.

**Text source:** https://s3.amazonaws.com/text-datasets/nietzsche.txt

The exact downloaded copy is stored at `data/nietzsche.txt`, so the experiment can also be reproduced offline.

| Characteristic | Value |
|---|---:|
| Characters | 598,861 |
| Word/punctuation tokens | 114,447 |
| Unique raw tokens | 10,237 |
| Approximate sentences | 3,530 |
| Mean sentence length | 28.56 tokens |
| Median sentence length | 22 tokens |
| Tokens occurring once | 4,950 |
| Final vocabulary size | 5,289 |
| Out-of-vocabulary rate | 4.33% |

## Text preprocessing and autoregressive sequence construction

The notebook performs the following reproducible pipeline:

1. Read the UTF-8 corpus, normalize Unicode and whitespace, and lowercase the text.
2. Tokenize words, numbers, and punctuation so punctuation remains learnable context.
3. Build the vocabulary from the training portion only, reserving special tokens such as `<UNK>`.
4. Preserve chronological order and split the 57,209 examples into 45,767 training, 5,721 validation, and 5,721 test examples. This prevents future passages from leaking into earlier partitions.
5. Construct sliding word-level sequences with a context length of 30 and a stride of 2. Each example maps
   `[w_t, ..., w_(t+29)]` to the next-token target `w_(t+30)`.
6. Batch the integer-encoded sequences with PyTorch `DataLoader` objects. Training batches are shuffled, whereas validation and test batches are deterministic.

## Models implemented

All four models share the same high-level pipeline: token embedding, recurrent sequence encoder, dropout regularization, and a linear vocabulary projection trained with cross-entropy loss.

- **Simple RNN:** a standard tanh recurrent network that provides a fast baseline but can struggle with long-range dependencies because of vanishing gradients.
- **Bidirectional RNN:** processes each fixed input context in forward and reverse directions before predicting the next word. It has a richer representation but more parameters; no target or future token beyond the supplied context is exposed.
- **LSTM:** uses input, forget, and output gates plus a cell state to preserve longer-range information.
- **GRU:** uses update and reset gates, offering gated memory with fewer operations than an LSTM.

## Experimental design, GPU training, and fine-tuning

The verified environment used Python 3.12.7, PyTorch 2.13.0+cu126, CUDA 12.6, cuDNN, and an NVIDIA GeForce RTX 3050 Laptop GPU. PyTorch's native `nn.RNN`, bidirectional `nn.RNN`, `nn.LSTM`, and `nn.GRU` modules dispatch to optimized CUDA/cuDNN kernels. The notebook records the selected device and includes a real forward/backward optimizer smoke test.

GPU-oriented training features include automatic mixed precision, fused AdamW where supported, pinned memory, non-blocking transfers, gradient clipping, label smoothing, learning-rate reduction, early stopping, and restoration of the best validation checkpoint.

Each architecture receives the same four-candidate, validation-only tuning budget. The search varies hidden-state size, dropout, learning rate, and weight decay. The test set is not used for model or hyperparameter selection. The winning configuration is reinitialized, trained with the full budget, and evaluated once on the test set. Architecture-specific winning dimensions are retained rather than forcing one configuration on every model.

| Model | Hidden size | Dropout | Learning rate | Weight decay |
|---|---:|---:|---:|---:|
| Simple RNN | 192 | 0.40 | 0.0015 | 0.0003 |
| Bidirectional RNN | 192 | 0.40 | 0.0015 | 0.0003 |
| LSTM | 256 | 0.30 | 0.0020 | 0.0001 |
| GRU | 256 | 0.30 | 0.0020 | 0.0001 |

## Dependencies

The pinned dependencies are listed in `requirements.txt`. Major packages include PyTorch with CUDA 12.6 support, NumPy, pandas, Matplotlib, scikit-learn, Requests, Jupyter, and nbconvert.

## Execution steps

From PowerShell, clone the public repository and create the Python 3.12 environment:

```powershell
git clone https://github.com/electro-glitch/ict4403-rnn-text-generation.git
cd ict4403-rnn-text-generation
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name fgai-python312-cuda --display-name "Python 3.12 (FGAI CUDA)"
python -m notebook ICT4403_Assignment2_Autoregressive_Text_Generation.ipynb
```

Select the **Python 3.12 (FGAI CUDA)** kernel and run **Kernel > Restart Kernel and Run All Cells**. CUDA is selected automatically when a compatible NVIDIA GPU is available, with a CPU fallback otherwise. The committed notebook is already executed, so its reported outputs can be inspected without retraining.

## Quantitative results

All reported values below come from the held-out test set. Lower loss and perplexity are better; higher accuracy and top-5 accuracy are better. Training time is measured for the final selected run on the stated machine.

| Model | Test loss | Perplexity | Accuracy | Top-5 accuracy | Training time (s) | Parameters (M) |
|---|---:|---:|---:|---:|---:|---:|
| **GRU** | **5.6738** | **291.15** | **14.65%** | 31.03% | 10.27 | 2.333 |
| LSTM | 5.6869 | 294.99 | 14.63% | **31.81%** | 12.08 | 2.432 |
| Simple RNN | 5.7150 | 303.37 | 14.19% | 30.96% | **6.70** | **1.760** |
| Bidirectional RNN | 5.7583 | 316.80 | 13.48% | 31.06% | 7.14 | 2.837 |

Additional plots compare learning curves, tuning improvements, accuracy, perplexity, parameter efficiency, throughput, generalization gap, and Pareto trade-offs. Machine-readable values are available in `artifacts/test_metrics.csv`, `artifacts/tuning_trials.csv`, `artifacts/tuning_impact.csv`, and `artifacts/additional_diagnostics.csv`.

## Model comparison and evaluation

- **GRU is the strongest overall model.** It achieves the lowest test loss and perplexity and the highest next-token accuracy, while training faster and using slightly fewer parameters than LSTM.
- **LSTM is a close second.** It obtains the best top-5 accuracy, suggesting that the correct continuation is often among its plausible alternatives, but its test perplexity and training cost are slightly higher than GRU's.
- **Simple RNN is the efficiency baseline.** It is the fastest and smallest model, but weaker perplexity and accuracy show the cost of having no gating mechanism for retaining context.
- **Bidirectional RNN does not justify its extra capacity here.** It has the largest parameter count and the poorest loss, perplexity, and top-1 accuracy. Reverse processing of an already-known context does not compensate for the lack of gated memory in this next-token task.
- **Qualitative inspection agrees with the numerical ranking.** The gated models generally preserve philosophical tone and local grammar better. GRU continuations most consistently balance context with varied wording; Simple RNN and BiRNN more often drift or reuse generic constructions. No compact model reliably sustains argument-level coherence over long continuations.

## Sampling-temperature analysis

Text is generated from the same three seed sequences for every model and temperature:

1. `the will to truth has led us`
2. `the philosopher of the future must`
3. `what is noble in the human being`

The experiment uses temperatures **0.5, 1.0, and 1.5**. Lower temperature sharpens the next-token distribution; higher temperature flattens it. The following values are averages across all models and prompts:

| Temperature | Distinct-1 | Distinct-2 | Repetition rate | Prompt overlap |
|---:|---:|---:|---:|---:|
| 0.5 | 0.44 | 0.74 | 0.26 | 0.08 |
| 1.0 | 0.81 | ~1.00 | ~0.00 | 0.05 |
| 1.5 | 0.95 | ~1.00 | ~0.00 | 0.02 |

- At **T = 0.5**, generation is conservative and locally more predictable, but its low diversity and 0.26 repetition rate reveal loops and repeated high-probability words. Coherence can improve sentence by sentence while originality declines.
- At **T = 1.0**, diversity increases sharply, repetition is almost eliminated, and the output usually retains more grammatical and contextual structure than at 1.5. This is the best overall operating point for readable yet non-deterministic samples.
- At **T = 1.5**, lexical diversity is highest and prompt overlap is lowest, but rare or weakly related choices increasingly damage grammatical continuity and philosophical coherence.

Distinct-1 and Distinct-2 measure unique unigrams and bigrams; repetition rate measures repeated generated tokens; prompt overlap estimates lexical consistency with the seed. These automatic measures are interpreted together with manual reading because diversity alone does not imply coherence. Full samples and per-run scores are stored in `artifacts/generations.csv` and `artifacts/generation_quality.csv`, and the comparison chart is `artifacts/temperature_analysis.png`.

## Conclusion

The **GRU provides the most effective autoregressive generation for this dataset**. It gives the best held-out loss, perplexity, and top-1 accuracy, remains more efficient than LSTM, and produces the most dependable balance of contextual consistency, diversity, and repetition in qualitative inspection. LSTM is a strong alternative when top-5 coverage is prioritized. Across architectures, **temperature 1.0** is the most useful default: 0.5 is too repetitive, while 1.5 gains diversity at a noticeable cost to coherence.

## Repository contents

```text
ict4403-rnn-text-generation/
|-- ICT4403_Assignment2_Autoregressive_Text_Generation.ipynb  # complete executed implementation
|-- README.md                                                   # setup, results, and analysis
|-- requirements.txt                                            # pinned dependencies
|-- data/
|   `-- nietzsche.txt                                           # dataset snapshot
`-- artifacts/
    |-- test_metrics.csv, tuning_trials.csv, tuned_hyperparameters.csv
    |-- tuning_impact.csv, additional_diagnostics.csv
    |-- generations.csv, generation_quality.csv
    |-- experiment_summary.json
    `-- *.png                                                    # comparison and diagnostic plots
```

Model checkpoints are recreated under `checkpoints/` when the notebook is run and are intentionally excluded from Git because they are generated binary files.

## Reproducibility notes and limitations

Random seeds, the chronological split, preprocessing settings, tuned configurations, and package versions are recorded in the notebook and artifacts. Exact wall-clock time can vary by GPU, driver, thermal state, and background load. Generated text is stochastic, so wording can vary unless the same random state is preserved. Finally, next-token accuracy is naturally modest for a 5,289-word vocabulary; perplexity, rank-based accuracy, diversity measures, and direct qualitative inspection therefore provide a more complete evaluation than accuracy alone.


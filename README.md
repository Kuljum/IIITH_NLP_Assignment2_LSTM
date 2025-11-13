# LSTM Language Model – Pride and Prejudice
IIIT Hyderabad – NLP Assignment 2

## Author
Name: kamalsahebgari kuljum
Roll Number: S201148
Institute: IIIT Hyderabad
Date: November 2025

## Overview
This project implements a word-level LSTM language model trained on Pride and Prejudice by Jane Austen using PyTorch.
The model predicts the next word in a sequence and demonstrates three learning behaviors:
1. Underfitting
2. Overfitting
3. Best Fit

The goal is to understand model capacity, generalization, and sequence modeling using LSTMs.

## Dataset
- Source: Project Gutenberg – Pride and Prejudice (Public Domain)
- Tokens Used: 30,000–60,000
- Vocabulary: 3,000–5,000 most frequent words

Preprocessing:
- Converted text to lowercase
- Removed punctuation and symbols
- Tokenized by whitespace
- Limited vocabulary to the most frequent words

## Model Architecture
| Component | Details |
|:--|:--|
| Embedding | 128-dimensional embeddings |
| LSTM | 2 layers, hidden size = 256 |
| Output Layer | Linear → Softmax |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr = 0.0005) |
| Sequence Length | 40–80 |
| Batch Size | 64 |
| Device | CPU / GPU |
| Gradient Clipping | Applied (max_norm = 1.0) |

## Experiments
| Experiment | Hidden Dim | Layers | Epochs | Observation |
|:--|:--|:--|:--|:--|
| Underfitting | 64 | 1 | 3 | High loss, poor learning |
| Overfitting | 512 | 3 | 8–10 | Training loss down, validation loss up |
| Best Fit | 256 | 2 | 15 | Stable and balanced learning |

## Results
| Model | Validation Loss | Perplexity | Observation |
|:--|:--|:--|:--|
| Underfit | ~9.7 | ~15000 | Model too simple |
| Overfit | ~9.2 | ~7000 | Memorized training data |
| Best Fit | 6.41 | 608.52 | Balanced and generalizing |

Generated Text Example:
“She was not sharpened me often to know and so very all in which stands whom he may offered nothing worse Lucas I do arguments you at last to Miss Watson was so lately for his he had she found out of a.”

## Extra Credit Work
- Gradient clipping for stable training
- Validation split and monitoring
- Vocabulary optimization
- Visualization of loss curves
- Temperature-based text generation
- CPU vs GPU comparison
- Underfit/Overfit/Best Fit experiments

## How to Run
1. Upload Pride_and_Prejudice-Jane_Austen.txt to your Colab workspace.
2. Open and run the notebook Assignment2_LSTM.ipynb.
3. (Optional) Switch to GPU: Runtime → Change runtime type → GPU.
4. Outputs will include:
   - Trained model (model_final.pth)
   - Loss plots (underfit_loss_curve.png, overfit_loss_curve.png, bestfit_loss_curve.png)
   - Generated text output

## Files Included
- Assignment2_LSTM.ipynb – Notebook
- README.md – Project documentation
- Assignment2_Final_Report.pdf – Report
- underfit_loss_curve.png
- overfit_loss_curve.png
- bestfit_loss_curve.png
- generated_text_output.png
- model_final.pth (optional)

## Understanding Checkpoints
This project demonstrates:
- Underfitting → low model capacity
- Overfitting → excessive model capacity
- Best Fit → balanced generalization

## References
- PyTorch Documentation – https://pytorch.org/docs
- Project Gutenberg: Pride and Prejudice by Jane Austen
- Jurafsky & Martin – Speech and Language Processing
- IIIT Hyderabad NLP Course (2025)

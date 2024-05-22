# Differentially Private Synthetic Data Generation using Large Language Models

## Project Overview

This project explores the integration of differential privacy with Large Language Models (LLMs) to generate synthetic data, particularly focusing on sensitive content like user chats. The methodology uses a heuristic approach to guide generation, maintaining utility and fidelity while being computationally efficient. The approach ensures context and semantic integrity, a critical aspect for sensitive data sources.

## Experimental Setup

### Dataset

-   **Dataset Used**: dair-ai/emotion dataset
    -   Comprises English Twitter messages labeled with six basic emotions: anger, fear, joy, love, sadness, and surprise.
    -   Two configurations: a split version with 20,000 examples divided into train, validation, and test splits, and an unsplit version with 416,809 examples.

### Model Configuration

-   **Models Used**: Bert-base-uncased and GPT2
    -   **GPT2**: Selected for its balance between computational efficiency and quality of synthetic data.
    -   **Bert-base-uncased**: Used for comparative analysis and semantic similarity computation.

### Preprocessing

-   **Named Entity Recognition (NER)**: Utilized the Bert-base-NER model to ensure the absence of personal or confidential information in the text inputs.
-   **Label Integration**: Appended labels to the actual text and incorporated special tokens.

### Differential Privacy Pipeline

-   **Process**:
    -   Dataset partitioned into minibatches.
    -   Forward pass: Data fed into the network layers.
    -   Backward pass: Loss function propagated in reverse, gradients clipped to a threshold, and noise added to ensure privacy.
    -   Updated model parameters with noisy, averaged, and clipped gradients.

### Fine-tuning and Text Generation

-   **Fine-tuning**:
    -   Tokenized and preprocessed dataset.
    -   Used dp-transformers library for integrating differential privacy into the training process.
    -   Trained and saved model weights and parameters.
-   **Text Generation**:
    -   Generated two sets of synthetic datasets: one with differential privacy and one without.
    -   Evaluated effects of differential privacy using varying epsilon (ε) settings.

### Evaluation Criteria

-   **Utility**: Measured by training a classifier on synthetic data and evaluating performance on the original test data.
-   **Fidelity**: Assessed using semantic similarity between original and synthetic data.
-   **Privacy**: Measured using the privacy accountant, calculating RDP (Renyi Differential Privacy) and PRV (Privacy Loss Variance).

### Utility Scores

| Epsilon (ε) | Utility Score |
|-------------|---------------|
| Benchmark   | 0.925         |
| ∞           | 0.861         |
| 16          | 0.652         |
| 8           | 0.592         |
| 3           | 0.569         |

### Semantic Similarity Scores

| Epsilon (ε) | Semantic Similarity |
|-------------|----------------------|
| ∞           | 0.511                |
| 16          | 0.428                |
| 8           | 0.407                |
| 3           | 0.322                |

### Generated Text Samples for Label 3 (Anger)

| Epsilon (ε) | Text Sample |
|-------------|-------------|
| Benchmark   | i am just so sick of feeling like this and i just want opinions please nothing rude and immature |
| ∞           | i am just feeling overwhelmed with all the things that i need to do in order to get to the point where i feel like i am going to be able to do what i want to do with my life |
| 16          | i am just feeling a little annoyed at myself for not being able to keep up with what is going on in my life and i feel like i am wasting my time and energy trying to figure out what is wrong with me and how to |
| 8           | i am just feeling a bit overwhelmed with all the things that i have to do to make my life a better place for myself and my family and i feel like i am in the wrong place at the wrong time in my life and i |
| 3           | i am just feeling a little dazed and confused about what to do and how to do it and i feel like i am wasting my time and energy trying to figure out what i should do and what i need to do to get there |



## Conclusion

This research validates that LLMs can be effectively conditioned to generate high-quality, differentially private synthetic data. The study demonstrates a trade-off between privacy and data quality, where higher privacy levels lead to diminished utility and fidelity. This approach has significant potential for applications requiring high data fidelity and utility while ensuring robust privacy protection.

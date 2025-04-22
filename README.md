# PhishFight++ Web Demo

An interactive web demonstration of the PhishFight++ system for generating and detecting phishing emails using adversarial machine learning techniques.

## Overview

PhishFight++ consists of two main components:

1. **PhishGen**: A phishing email generator that uses language models like GPT-2 and T5 to create realistic phishing emails with various attack types and themes.

2. **PhishShield**: A phishing detection system that uses transformer models like BERT and DistilBERT to classify emails as legitimate or phishing.

The system uses an adversarial training approach where the defender models continuously improve based on the attacker's evolving tactics.

## Features

- Generate realistic phishing emails with different attack patterns and themes
- Classify emails as phishing or legitimate with confidence scores
- Understand why an email was classified as phishing using LIME explainability
- Track model performance metrics over time
- Interactive web UI for security researchers and educators

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Running the Demo

1. Start the Flask API server:

```bash
python app.py
```

2. In a separate terminal, start the Gradio web interface:

```bash
python gradio_app.py
```

3. Access the web interface at http://localhost:7860

## System Requirements

- Python 3.8+
- CUDA-capable GPU recommended for faster model inference
- At least 8GB RAM (16GB recommended)
- At least ca. 5GB disk space for model weights

## UI Components

### Attack Simulation
Generate phishing emails by selecting:
- Attack type (standard, prompt injection, semantic perturbation, style mimicry)
- Theme (banking, IRS, COVID-19, tech support, shipping)
- Number of emails to generate

### Defense Classification
Classify emails as phishing or legitimate:
- Input your own text or use generated examples
- View confidence scores from individual models or ensemble
- Visualize classification confidence

### Explainability
Understand model decisions:
- Visualize which words/phrases influenced the classification
- Compare explanations from different models

### Evaluation Dashboard
Track model performance:
- View accuracy, precision, recall, F1 score, AUC, and evasion rate
- See how performance evolves over adversarial training iterations

## Note on Production Use

This is a demonstration system. For real-world deployment, consider:
- Training on larger, more diverse datasets
- Regular model updates and fine-tuning
- Integration with email security infrastructure
- Human-in-the-loop review processes

## License

This project is for educational and research purposes only. Do not use the generated phishing emails for malicious purposes.

## Attribution

This project builds upon the PhishFight++ codebase, leveraging transformer models and adversarial machine learning techniques for phishing email generation and detection.

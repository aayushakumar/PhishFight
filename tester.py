import os
import torch
import numpy as np
import pandas as pd
import random
import re
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from umap import UMAP
from collections import Counter
import gc
import uuid
from datetime import datetime
import transformers
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
    BertForSequenceClassification, RobertaForSequenceClassification,
    DistilBertForSequenceClassification, DebertaForSequenceClassification,
    TrainingArguments, Trainer
)
import datasets
from datasets import Dataset, DatasetDict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import lime
import lime.lime_text
import shap

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
config = {
    "data": {
        "synthetic_data_size": 5000,
        "real_data_ratio": 0.3,
        "test_size": 0.2,
        "val_size": 0.1,
    },
    "attacker": {
        "models": ["gpt2", "gpt-neox", "llama-2", "t5"],
        "attack_types": ["standard", "prompt_injection", "semantic_perturbation", "style_mimicry"],
        "themes": ["banking", "irs", "covid-19", "tech_support", "shipping"],
        "batch_size": 16,
        "max_length": 512,
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 40,
    },
    "defender": {
        "models": ["bert", "roberta", "distilbert", "deberta"],
        "classifiers": ["transformer", "one-class-svm", "logistic-regression", "lstm"],
        "ensemble_methods": ["majority", "confidence"],
        "batch_size": 32,
        "max_length": 512,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "epochs": 5,
    },
    "adversarial_training": {
        "iterations": 5,
        "hard_sample_count": 5,
        "augmentation_types": ["synonym", "homoglyph", "typo"],
        "augmentation_prob": 0.3,
    },
    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1", "auc", "evasion_rate"],
        "visualizations": ["confusion_matrix", "tsne", "umap", "f1_evolution"],
    }
}

# Generate run ID and timestamp
run_id = str(uuid.uuid4())
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
config["run_id"] = run_id
config["timestamp"] = timestamp

# Save config to JSON
config_path = f"config_{run_id}_{timestamp}.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f"Config saved to {config_path}")

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_dataset(config):
    print("Loading datasets...")
    real_phishing_data = []
    real_legitimate_data = []
    
    try:
        nazario_path = "../input/nazario-phishing-corpus/phishing"
        if os.path.exists(nazario_path):
            nazario_files = os.listdir(nazario_path)
            print(f"Found {len(nazario_files)} files in Nazario corpus")
            for filename in tqdm(nazario_files[:1000]):
                with open(os.path.join(nazario_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    real_phishing_data.append({"text": content, "source": "nazario", "label": 1})
    except Exception as e:
        print(f"Could not load Nazario corpus: {e}")
    
    try:
        phishtank_path = "../input/phishtank-data/phishtank.csv"
        if os.path.exists(phishtank_path):
            phishtank_df = pd.read_csv(phishtank_path)
            print(f"Found {len(phishtank_df)} entries in PhishTank data")
            for _, row in tqdm(phishtank_df.iterrows()):
                if 'content' in row:
                    real_phishing_data.append({"text": row['content'], "source": "phishtank", "label": 1})
    except Exception as e:
        print(f"Could not load PhishTank data: {e}")
    
    try:
        enron_path = "../input/enron-email-dataset/emails.csv"
        if os.path.exists(enron_path):
            enron_df = pd.read_csv(enron_path)
            print(f"Found {len(enron_df)} entries in Enron dataset")
            for _, row in tqdm(enron_df.iterrows()):
                if 'message' in row:
                    real_legitimate_data.append({"text": row['message'], "source": "enron", "label": 0})
    except Exception as e:
        print(f"Could not load Enron data: {e}")
    
    try:
        uci_path = "../input/uci-email-features/emails.csv"
        if os.path.exists(uci_path):
            uci_df = pd.read_csv(uci_path)
            print(f"Found {len(uci_df)} entries in UCI dataset")
            for _, row in tqdm(uci_df.iterrows()):
                if 'content' in row and 'label' in row:
                    item = {"text": row['content'], "source": "uci", "label": int(row['label'])}
                    if item["label"] == 1:
                        real_phishing_data.append(item)
                    else:
                        real_legitimate_data.append(item)
    except Exception as e:
        print(f"Could not load UCI data: {e}")
    
    print(f"Loaded {len(real_phishing_data)} real phishing emails and {len(real_legitimate_data)} legitimate emails")
    
    if len(real_phishing_data) < 100 or len(real_legitimate_data) < 100:
        print("Not enough real data found, creating synthetic dataset")
        phishing_templates = [
            "Dear {user}, your account has been compromised. Please click {link} to verify your information.",
            "URGENT: Your {service} account needs verification. Update your details here: {link}",
            "Security Alert: Unusual activity detected on your {service} account. Please verify your identity: {link}",
        ]
        services = ["PayPal", "Apple", "Amazon", "Bank of America"]
        domains = ["security-check.com", "account-verify.net"]
        
        synthetic_phishing = []
        for _ in range(1000):
            template = random.choice(phishing_templates)
            service = random.choice(services)
            domain = random.choice(domains)
            link = f"https://{service.lower()}-{random.randint(100, 999)}.{domain}"
            user = f"user{random.randint(1000, 9999)}"
            email = template.format(user=user, service=service, link=link)
            synthetic_phishing.append({"text": email, "source": "synthetic", "label": 1})
        
        legitimate_templates = [
            "Hi {name}, just following up on our meeting yesterday. Can you send me those reports?",
            "Thank you for your recent purchase from {store}. Your order #{order_number} has shipped.",
        ]
        names = ["Alex", "Taylor", "Jordan"]
        stores = ["Office Supply Co.", "Tech Gadgets Inc."]
        
        synthetic_legitimate = []
        for _ in range(1000):
            template = random.choice(legitimate_templates)
            name = random.choice(names)
            store = random.choice(stores)
            order_number = f"{random.randint(10000, 99999)}"
            email = template.format(name=name, store=store, order_number=order_number, service=random.choice(services))
            synthetic_legitimate.append({"text": email, "source": "synthetic", "label": 0})
        
        all_phishing = real_phishing_data + synthetic_phishing
        all_legitimate = real_legitimate_data + synthetic_legitimate
    else:
        all_phishing = real_phishing_data
        all_legitimate = real_legitimate_data
    
    all_data = all_phishing + all_legitimate
    random.shuffle(all_data)
    df = pd.DataFrame(all_data)
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    
    # Integrity checks
    assert all(isinstance(t, str) and len(t) > 0 for t in df['text']), "Some texts are empty or not strings"
    assert all(l in [0, 1] for l in df['label']), "Labels must be 0 or 1"
    
    # Stratified sampling and logging
    print("Label distribution before split:", dict(Counter(df['label'])))
    train_df, test_df = train_test_split(df, test_size=config['data']['test_size'], stratify=df['label'], random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=config['data']['val_size']/(1-config['data']['test_size']), 
                                        stratify=train_df['label'], random_state=SEED)
    
    print("Train label distribution:", dict(Counter(train_df['label'])))
    print("Validation label distribution:", dict(Counter(val_df['label'])))
    print("Test label distribution:", dict(Counter(test_df['label'])))
    
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    print(f"Dataset splits: Train {len(dataset_dict['train'])}, Val {len(dataset_dict['validation'])}, Test {len(dataset_dict['test'])}")
    return dataset_dict

class PhishGen:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.loaded_models = []
        print("Initializing PhishGen attacker models...")
        
        for model_name in config["attacker"]["models"]:
            try:
                if model_name == "gpt2":
                    self.tokenizers["gpt2"] = AutoTokenizer.from_pretrained("gpt2")
                    self.models["gpt2"] = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
                    self.loaded_models.append("gpt2")
                    print("Loaded GPT-2 model")
                elif model_name == "gpt-neox":
                    self.tokenizers["gpt-neox"] = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
                    self.models["gpt-neox"] = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
                    self.loaded_models.append("gpt-neox")
                    print("Loaded GPT-Neo 1.3B model")
                elif model_name == "llama-2":
                    self.tokenizers["llama-2"] = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                    self.models["llama-2"] = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
                    self.loaded_models.append("llama-2")
                    print("Loaded TinyLlama model")
                elif model_name == "t5":
                    self.tokenizers["t5"] = AutoTokenizer.from_pretrained("google-t5/t5-small")
                    self.models["t5"] = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").to(device)
                    self.loaded_models.append("t5")
                    print("Loaded T5-small model")
            except Exception as e:
                print(f"Could not load {model_name}: {e}")
        
        self.use_fallback = not self.loaded_models
        if self.use_fallback:
            print("Using fallback template-based generator")

    def _create_prompt(self, attack_type, theme=None):
        base_prompt = "Generate a phishing email "
        if theme:
            base_prompt += f"related to {theme} "
        if attack_type == "standard":
            base_prompt += "to steal credentials."
        elif attack_type == "prompt_injection":
            base_prompt += "with subtle manipulation."
        elif attack_type == "semantic_perturbation":
            base_prompt += "using unusual wording."
        elif attack_type == "style_mimicry":
            companies = {"banking": "Bank of America", "irs": "IRS", "covid-19": "CDC", "tech_support": "Microsoft", "shipping": "Amazon"}
            company = companies.get(theme, "a company")
            base_prompt += f"mimicking {company} style."
        return base_prompt

    def _fallback_generate(self, attack_type, theme):
        templates = {
            "standard": "Dear customer, your {theme} account is at risk. Verify here: {link}",
            "prompt_injection": "Urgent {theme} update required. Confirm details: {link}",
            "semantic_perturbation": "Account {theme} needs verify urgent. Click: {link}",
            "style_mimicry": "Notice from {theme}: Action needed at {link}"
        }
        domains = ["secure-login.com", "verify-now.net"]
        link = f"https://{random.choice(domains)}/{random.randint(1000, 9999)}"
        email = templates.get(attack_type, templates["standard"]).replace("{theme}", theme or "service").replace("{link}", link)
        return email + "\n\nSupport Team"

    def generate_phishing_emails(self, num_samples=10, attack_type=None, theme=None):
        attack_type = attack_type or random.choice(self.config["attacker"]["attack_types"])
        theme = theme or random.choice(self.config["attacker"]["themes"])
        results = []
        
        for _ in tqdm(range(num_samples), desc=f"Generating {attack_type} emails"):
            if self.use_fallback:
                email = self._fallback_generate(attack_type, theme)
            else:
                model_name = random.choice(self.loaded_models)
                model = self.models[model_name]
                tokenizer = self.tokenizers[model_name]
                prompt = self._create_prompt(attack_type, theme)
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]
                
                try:
                    if model_name == "t5":
                        outputs = model.generate(
                            **inputs,
                            max_length=self.config["attacker"]["max_length"],
                            temperature=self.config["attacker"]["temperature"],
                            top_p=self.config["attacker"]["top_p"],
                            top_k=self.config["attacker"]["top_k"],
                            do_sample=True
                        )
                        email = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        outputs = model.generate(
                            **inputs,
                            max_length=self.config["attacker"]["max_length"],
                            temperature=self.config["attacker"]["temperature"],
                            top_p=self.config["attacker"]["top_p"],
                            top_k=self.config["attacker"]["top_k"],
                            do_sample=True,
                            return_dict_in_generate=True,
                            output_scores=False
                        )
                        email = tokenizer.decode(outputs.sequences[0][input_length:], skip_special_tokens=True)
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    email = self._fallback_generate(attack_type, theme)
            
            assert len(email) > 0, f"Generated email is empty for {model_name}"
            results.append({"text": email, "model": model_name if not self.use_fallback else "fallback", 
                           "attack_type": attack_type, "theme": theme, "label": 1, "prompt": prompt})
        return results

    def apply_adversarial_perturbations(self, text, perturbation_type):
        if perturbation_type == "synonym":
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            keywords = ["account", "verify", "login", "secure", "urgent"]
            for i, (word, pos) in enumerate(pos_tags):
                if word.lower() in keywords and random.random() < 0.2:
                    synonyms = [lemma.name().replace('_', ' ') for syn in wordnet.synsets(word) for lemma in syn.lemmas()]
                    if synonyms:
                        words[i] = random.choice(synonyms)
            return ' '.join(words)
        
        elif perturbation_type == "homoglyph":
            homoglyphs = {'a': 'а', 'e': 'е', 'i': 'і', 'o': 'о', 'p': 'р', 's': 'ѕ'}
            result = ""
            for char in text:
                if char.lower() in homoglyphs and random.random() < 0.15:
                    result += homoglyphs[char.lower()] if char.islower() else homoglyphs[char.lower()].upper()
                else:
                    result += char
            return result
        
        elif perturbation_type == "typo":
            words = text.split()
            for i in range(len(words)):
                if random.random() < 0.15 and len(words[i]) > 2:
                    pos = random.randint(0, len(words[i]) - 2)
                    words[i] = words[i][:pos] + words[i][pos+1] + words[i][pos] + words[i][pos+2:]
            return ' '.join(words)
        
        return text

    def condition_on_real_corpus(self, real_examples, num_samples=10, attack_type=None, theme=None):
        if not real_examples:
            return self.generate_phishing_emails(num_samples, attack_type, theme)
        
        results = []
        examples = random.sample(real_examples, min(num_samples, len(real_examples)))
        
        for example in tqdm(examples, desc="Generating conditioned emails"):
            text = example.get('text', '')
            if not text or len(text) < 10:
                continue
                
            keywords = [w for w in text.split() if len(w) > 4 and w.lower() not in ["the", "and", "for"]]
            keywords = random.sample(keywords, min(5, len(keywords))) if keywords else []
            prompt = f"Generate a phishing email with keywords: {', '.join(keywords)}."
            
            if self.use_fallback:
                email = self._fallback_generate(attack_type or "standard", theme or "mixed")
            else:
                model_name = random.choice(self.loaded_models)
                model = self.models[model_name]
                tokenizer = self.tokenizers[model_name]
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]
                
                try:
                    if model_name == "t5":
                        email = tokenizer.decode(model.generate(**inputs, max_length=512, do_sample=True)[0], skip_special_tokens=True)
                    else:
                        email = tokenizer.decode(model.generate(**inputs, max_length=512, do_sample=True).sequences[0][input_length:], skip_special_tokens=True)
                except:
                    email = self._fallback_generate(attack_type or "standard", theme or "mixed")
            
            if random.random() < 0.3:
                perturbation = random.choice(self.config["adversarial_training"]["augmentation_types"])
                email = self.apply_adversarial_perturbations(email, perturbation)
            
            results.append({"text": email, "model": model_name if not self.use_fallback else "fallback", 
                           "attack_type": attack_type or "conditioned", "theme": theme or "mixed", 
                           "conditioned_on": example.get('source', 'unknown'), "label": 1})
        return results

    def adversarial_fine_tuning(self, examples, rewards):
        if self.use_fallback or not examples or len(rewards) != len(examples):
            return False
        try:
            from trl import PPOTrainer, PPOConfig
            model_name = "gpt2" if "gpt2" in self.loaded_models else self.loaded_models[0]
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            ppo_config = PPOConfig(batch_size=4, learning_rate=5e-5, horizon=512)
            ppo_trainer = PPOTrainer(model=model, config=ppo_config, tokenizer=tokenizer)
            
            for ex, reward in zip(examples, rewards):
                prompt = ex.get("prompt", "Generate a phishing email.")  # Default prompt if missing
                query = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
                response = tokenizer(ex["text"], return_tensors="pt").to(device)["input_ids"]
                ppo_trainer.step(query, response, [reward])
            
            self.models[model_name] = model
            return True
        except ImportError:
            print("PPOTrainer not available, falling back to standard fine-tuning")
            model_name = "gpt2" if "gpt2" in self.loaded_models else self.loaded_models[0]
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            
            train_dataset = Dataset.from_pandas(pd.DataFrame([{"text": ex["text"]} for ex in examples])).map(tokenize_function, batched=True)
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                learning_rate=5e-5,
                save_strategy="no",
            )
            
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
            trainer.train()
            self.models[model_name] = model
            return True

class PhishShield:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.classifiers = {}
        self.loaded_models = []
        self.explainers = {}
        self.shap_cache = {}  # Cache for SHAP outputs
        
        print("Initializing PhishShield defender models...")
        for model_name in config["defender"]["models"]:
            try:
                if model_name == "bert":
                    self.tokenizers["bert"] = AutoTokenizer.from_pretrained("bert-base-uncased")
                    self.models["bert"] = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
                    self.loaded_models.append("bert")
                    print("Loaded BERT model")
                elif model_name == "roberta":
                    self.tokenizers["roberta"] = AutoTokenizer.from_pretrained("roberta-base")
                    self.models["roberta"] = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
                    self.loaded_models.append("roberta")
                    print("Loaded RoBERTa model")
                elif model_name == "distilbert":
                    self.tokenizers["distilbert"] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                    self.models["distilbert"] = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
                    self.loaded_models.append("distilbert")
                    print("Loaded DistilBERT model")
                elif model_name == "deberta":
                    self.tokenizers["deberta"] = AutoTokenizer.from_pretrained("microsoft/deberta-base")
                    self.models["deberta"] = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2).to(device)
                    self.loaded_models.append("deberta")
                    print("Loaded DeBERTa model")
            except Exception as e:
                print(f"Could not load {model_name}: {e}")
        
        if "lime" in globals():
            self.explainers["lime"] = lime.lime_text.LimeTextExplainer(class_names=["Legitimate", "Phishing"])
            print("Loaded LIME explainer")
        if "shap" in globals() and self.loaded_models:
            self.explainers["shap"] = True
            print("SHAP explainer available")

    def _tokenize_text(self, texts, model_name):
        tokenizer = self.tokenizers.get(model_name)
        if not tokenizer:
            raise ValueError(f"Tokenizer for {model_name} not found")
        texts = [texts] if isinstance(texts, str) else texts
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    def _predict_transformer(self, texts, model_name):
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        inputs = self._tokenize_text(texts, model_name)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range"
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-5), "Probabilities do not sum to 1"
            return probs

    def train_model(self, dataset, model_name="bert", classifier_type="transformer"):
        print(f"Training {model_name} ({classifier_type})")
        
        if classifier_type == "transformer" and model_name in self.loaded_models:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            columns = ["input_ids", "attention_mask", "label"]
            if "token_type_ids" in tokenized_dataset["train"].column_names:
                columns.append("token_type_ids")
            tokenized_dataset = tokenized_dataset.remove_columns([c for c in tokenized_dataset["train"].column_names if c not in columns])
            tokenized_dataset.set_format("torch")
            
            training_args = TrainingArguments(
                output_dir=f"./results_{model_name}",
                num_train_epochs=self.config["defender"]["epochs"],
                per_device_train_batch_size=self.config["defender"]["batch_size"],
                learning_rate=self.config["defender"]["learning_rate"],
                weight_decay=self.config["defender"]["weight_decay"],
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                fp16=True if torch.cuda.is_available() else False,  # Mixed-precision
            )
            
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=1)
                return {
                    "accuracy": accuracy_score(labels, predictions),
                    "precision": precision_score(labels, predictions),
                    "recall": recall_score(labels, predictions),
                    "f1": f1_score(labels, predictions)
                }
            
            trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"], 
                            eval_dataset=tokenized_dataset["validation"], compute_metrics=compute_metrics)
            trainer.train()
            eval_results = trainer.evaluate()
            print(f"Evaluation results: {eval_results}")
            self.models[model_name] = model
            return model
        
        elif classifier_type == "one-class-svm":
            from sklearn.feature_extraction.text import TfidfVectorizer
            legitimate_texts = [ex["text"] for ex in dataset["train"] if ex["label"] == 0]
            if not legitimate_texts:
                return None
            vectorizer = TfidfVectorizer(max_features=10000)
            X_train = vectorizer.fit_transform(legitimate_texts)
            model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            model.fit(X_train)
            self.classifiers[f"{model_name}_svm"] = {"model": model, "vectorizer": vectorizer}
            return model
        
        elif classifier_type == "logistic-regression":
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [ex["text"] for ex in dataset["train"]]
            labels = [ex["label"] for ex in dataset["train"]]
            vectorizer = TfidfVectorizer(max_features=10000)
            X_train = vectorizer.fit_transform(texts)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, labels)
            self.classifiers[f"{model_name}_lr"] = {"model": model, "vectorizer": vectorizer}
            return model
        
        elif classifier_type == "lstm":
            texts = [ex["text"] for ex in dataset["train"]]
            labels = [ex["label"] for ex in dataset["train"]]
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            data = pad_sequences(sequences, maxlen=200)
            labels = np.array(labels)
            
            model = Sequential([
                Embedding(len(tokenizer.word_index) + 1, 100, input_length=200),
                Bidirectional(LSTM(64)),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(data, labels, batch_size=32, epochs=5, validation_split=0.2)
            self.classifiers[f"{model_name}_lstm"] = {"model": model, "tokenizer": tokenizer, "max_length": 200}
            return model
        return None

    def predict(self, texts, model_name=None, classifier_type=None, return_raw=False):
        texts = [texts] if isinstance(texts, str) else texts
        if not model_name and not classifier_type:
            return self.ensemble_predict(texts, return_type="pred_conf")
        
        if classifier_type == "transformer" or (classifier_type is None and model_name in self.loaded_models):
            model_name = model_name or self.loaded_models[0]
            probs = self._predict_transformer(texts, model_name)
            if return_raw:
                return probs
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            return list(zip(preds.tolist(), confs.tolist()))
        
        classifier_key = f"{model_name}_{classifier_type.split('-')[0]}"
        if classifier_key not in self.classifiers:
            return self.ensemble_predict(texts, return_type="pred_conf")
        
        classifier = self.classifiers[classifier_key]
        if classifier_type == "one-class-svm":
            X = classifier["vectorizer"].transform(texts)
            preds = [0 if p == 1 else 1 for p in classifier["model"].predict(X)]
            scores = classifier["model"].decision_function(X)
            confs = [1 - 1/(1 + np.exp(abs(s))) for s in scores]
            if return_raw:
                return np.column_stack(([1-c for c in confs], confs))
            return list(zip(preds, confs))
        
        elif classifier_type == "logistic-regression":
            X = classifier["vectorizer"].transform(texts)
            probs = classifier["model"].predict_proba(X)
            if return_raw:
                return probs
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            return list(zip(preds.tolist(), confs.tolist()))
        
        elif classifier_type == "lstm":
            sequences = classifier["tokenizer"].texts_to_sequences(texts)
            data = pad_sequences(sequences, maxlen=classifier["max_length"])
            probs = classifier["model"].predict(data, verbose=0).flatten()
            preds = [1 if p > 0.5 else 0 for p in probs]
            confs = [p if p > 0.5 else 1-p for p in probs]
            if return_raw:
                return np.column_stack((1-probs, probs))
            return list(zip(preds, confs))
        return None

    def ensemble_predict(self, texts, ensemble_method="majority", return_type="pred_conf"):
        texts = [texts] if isinstance(texts, str) else texts
        all_preds = []
        all_probs = []
        
        for model_name in self.loaded_models:
            try:
                probs = self._predict_transformer(texts, model_name)
                all_preds.append(np.argmax(probs, axis=1))
                all_probs.append(probs)
            except:
                continue
        
        for key, clf in self.classifiers.items():
            try:
                if "svm" in key:
                    X = clf["vectorizer"].transform(texts)
                    preds = np.array([0 if p == 1 else 1 for p in clf["model"].predict(X)])
                    scores = clf["model"].decision_function(X)
                    probs = np.array([[1/(1+np.exp(-s)), 1-1/(1+np.exp(-s))] for s in scores])
                elif "lr" in key:
                    X = clf["vectorizer"].transform(texts)
                    probs = clf["model"].predict_proba(X)
                    preds = np.argmax(probs, axis=1)
                elif "lstm" in key:
                    sequences = clf["tokenizer"].texts_to_sequences(texts)
                    data = pad_sequences(sequences, maxlen=clf["max_length"])
                    probs_raw = clf["model"].predict(data, verbose=0).flatten()
                    preds = np.array([1 if p > 0.5 else 0 for p in probs_raw])
                    probs = np.column_stack((1-probs_raw, probs_raw))
                all_preds.append(preds)
                all_probs.append(probs)
            except:
                continue
        
        if not all_preds:
            return [(0, 0.5) for _ in texts]
        
        if ensemble_method == "majority":
            stacked_preds = np.stack(all_preds, axis=0)
            preds = [Counter(stacked_preds[:, i]).most_common(1)[0][0] for i in range(len(texts))]
            confs = [sum(stacked_preds[:, i] == p) / len(all_preds) for i, p in enumerate(preds)]
            if return_type == "probs":
                avg_probs = np.mean(all_probs, axis=0)
                return avg_probs
            return list(zip(preds, confs))
        
        elif ensemble_method == "confidence":
            stacked_probs = np.stack(all_probs, axis=0)
            avg_probs = np.mean(stacked_probs, axis=0)
            if return_type == "probs":
                return avg_probs
            preds = np.argmax(avg_probs, axis=1)
            confs = np.max(avg_probs, axis=1)
            return list(zip(preds.tolist(), confs.tolist()))
        return [(0, 0.5) for _ in texts]

    def explain_prediction(self, text, model_name="bert", method="lime"):
        if method == "lime" and "lime" in self.explainers:
            def predict_fn(texts):
                return self._predict_transformer(texts, model_name)
            explanation = self.explainers["lime"].explain_instance(text, predict_fn, num_features=10)
            return explanation.as_list()
        
        elif method == "shap" and "shap" in self.explainers and model_name in self.loaded_models:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            inputs = self._tokenize_text([text], model_name)
            
            cache_key = f"{model_name}_{text[:50]}"
            if cache_key in self.shap_cache:
                return self.shap_cache[cache_key]
            
            def predict_fn(input_ids):
                with torch.no_grad():
                    return model(input_ids=torch.tensor(input_ids).to(device)).logits.cpu().numpy()
            
            explainer = shap.Explainer(predict_fn, tokenizer)
            shap_values = explainer(inputs["input_ids"].cpu().numpy())
            self.shap_cache[cache_key] = shap_values
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, feature_names=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), show=False)
            plt.savefig(f"shap_summary_{model_name}.png")
            plt.close()
            return shap_values

def train_defenders(phishshield, dataset):
    for model_name in phishshield.loaded_models:
        phishshield.train_model(dataset, model_name, "transformer")
    for clf_type in ["one-class-svm", "logistic-regression", "lstm"]:
        phishshield.train_model(dataset, "bert", clf_type)

def generate_adversaries(phishgen, num_samples=50):
    phishing_emails = phishgen.generate_phishing_emails(num_samples)
    for email in phishing_emails:
        if random.random() < config["adversarial_training"]["augmentation_prob"]:
            pert = random.choice(config["adversarial_training"]["augmentation_types"])
            email["text"] = phishgen.apply_adversarial_perturbations(email["text"], pert)
    return phishing_emails

def evaluate_metrics(phishshield, test_texts, test_labels, adversarial_texts):
    all_texts = test_texts + adversarial_texts
    all_labels = test_labels + [1] * len(adversarial_texts)
    probs = phishshield.ensemble_predict(all_texts, ensemble_method="confidence", return_type="probs")
    all_preds = np.argmax(probs, axis=1)
    all_confs = np.max(probs, axis=1)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, probs[:, 1]),
        "evasion_rate": 1 - recall_score([1] * len(adversarial_texts), all_preds[-len(adversarial_texts):])
    }
    
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    metrics["class_0"] = class_report["0"]
    metrics["class_1"] = class_report["1"]
    
    fp_indices = [i for i in range(len(all_labels)) if all_labels[i] == 0 and all_preds[i] == 1]
    fn_indices = [i for i in range(len(all_labels)) if all_labels[i] == 1 and all_preds[i] == 0]
    fp_examples = [all_texts[i] for i in fp_indices[:5]]
    fn_examples = [all_texts[i] for i in fn_indices[:5]]
    
    return metrics, all_preds, all_confs, fp_examples, fn_examples

def visualize_metrics(metrics, iteration, all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Iteration {iteration+1}")
    plt.savefig(f"confusion_matrix_iter{iteration+1}.png")
    plt.close()

def evaluate_phishfight(dataset, phishgen, phishshield):
    print("Starting evaluation...")
    metrics_history = {"f1": []}
    
    test_texts = [ex["text"] for ex in dataset["test"]]
    test_labels = [ex["label"] for ex in dataset["test"]]
    
    for iteration in range(config["adversarial_training"]["iterations"]):
        print(f"Iteration {iteration + 1}/{config['adversarial_training']['iterations']}")
        
        train_defenders(phishshield, dataset)
        
        adversarial_emails = generate_adversaries(phishgen)
        
        metrics, all_preds, all_confs, fp_examples, fn_examples = evaluate_metrics(
            phishshield, test_texts, test_labels, [e["text"] for e in adversarial_emails])
        metrics_history["f1"].append(metrics["f1"])
        print(f"Metrics: {metrics}")
        print("False Positives:", fp_examples)
        print("False Negatives:", fn_examples)
        with open(f"metrics_log_iter{iteration+1}.json", 'w') as f:
            json.dump({"metrics": metrics, "fp_examples": fp_examples, "fn_examples": fn_examples}, f, indent=4)
        
        visualize_metrics(metrics, iteration, test_labels + [1] * len(adversarial_emails), all_preds)
        
        probs = phishshield.ensemble_predict([e["text"] for e in adversarial_emails], 
                                            ensemble_method="confidence", return_type="probs")
        rewards = probs[:, 0]  # P(class=0) as reward
        phishgen.adversarial_fine_tuning(adversarial_emails, rewards)
        
        # Memory management
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics_history["f1"]) + 1), metrics_history["f1"], marker='o')
    plt.title("F1-Score Evolution Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("F1-Score")
    plt.savefig("f1_evolution.png")
    plt.close()
    
    # SHAP visualization for a sample
    sample_text = test_texts[0]
    shap_values = phishshield.explain_prediction(sample_text, method="shap")

if __name__ == "__main__":
    dataset = load_dataset(config)
    phishgen = PhishGen(config)
    phishshield = PhishShield(config)
    evaluate_phishfight(dataset, phishgen, phishshield)

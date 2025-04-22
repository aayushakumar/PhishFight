import gradio as gr
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from PIL import Image
from io import BytesIO

# API endpoint (change this to your actual deployment URL)
API_URL = "http://localhost:5000"

# Get configuration from API
try:
    config = requests.get(f"{API_URL}/config").json()["config"]
except:
    # Fallback config if API is not running
    config = {
        "attacker": {
            "attack_types": ["standard", "prompt_injection", "semantic_perturbation", "style_mimicry"],
            "themes": ["banking", "irs", "covid-19", "tech_support", "shipping"]
        }
    }

def generate_phishing_emails(attack_type, theme, num_samples):
    try:
        response = requests.post(f"{API_URL}/generate", json={
            "attack_type": attack_type, 
            "theme": theme, 
            "num_samples": num_samples
        })
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                emails = data["emails"]
                df = pd.DataFrame(emails)
                return df, "Emails generated successfully!"
            else:
                return None, f"Error: {data.get('error', 'Unknown error')}"
        else:
            return None, f"API Error: HTTP {response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def classify_email(email_text, use_ensemble=True):
    try:
        response = requests.post(f"{API_URL}/classify", json={
            "text": email_text,
            "model_name": None if use_ensemble else "bert"  # Use ensemble if True, else use BERT
        })
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                result = data["result"][0]  # First result (could be multiple in batch mode)
                
                # Create classification result text
                if use_ensemble:
                    ensemble_pred = "PHISHING" if result["ensemble_prediction"] == 1 else "LEGITIMATE"
                    ensemble_conf = result["ensemble_confidence"] * 100
                    result_text = f"Ensemble prediction: {ensemble_pred} (Confidence: {ensemble_conf:.1f}%)\n\n"
                    
                    # Add individual model results
                    result_text += "Individual model predictions:\n"
                    for model_result in result["model_results"]:
                        model_pred = "PHISHING" if model_result["prediction"] == 1 else "LEGITIMATE"
                        model_conf = model_result["confidence"] * 100
                        result_text += f"- {model_result['model'].upper()}: {model_pred} ({model_conf:.1f}%)\n"
                else:
                    pred = "PHISHING" if result["prediction"] == 1 else "LEGITIMATE"
                    conf = result["confidence"] * 100
                    result_text = f"BERT prediction: {pred} (Confidence: {conf:.1f}%)"
                
                # Create bar chart for confidence visualization
                if use_ensemble:
                    models = [model_result["model"].upper() for model_result in result["model_results"]]
                    models.append("ENSEMBLE")
                    
                    phishing_probs = [model_result["probabilities"]["phishing"] * 100 
                                     for model_result in result["model_results"]]
                    phishing_probs.append(result["ensemble_probabilities"]["phishing"] * 100)
                    
                    legitimate_probs = [model_result["probabilities"]["legitimate"] * 100 
                                       for model_result in result["model_results"]]
                    legitimate_probs.append(result["ensemble_probabilities"]["legitimate"] * 100)
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Legitimate', x=models, y=legitimate_probs, marker_color='green'),
                        go.Bar(name='Phishing', x=models, y=phishing_probs, marker_color='red')
                    ])
                    fig.update_layout(
                        title="Confidence Scores by Model",
                        xaxis_title="Model",
                        yaxis_title="Confidence (%)",
                        barmode='group',
                        height=400
                    )
                else:
                    fig = go.Figure(data=[
                        go.Bar(name='Legitimate', x=["BERT"], y=[result["probabilities"]["legitimate"] * 100], marker_color='green'),
                        go.Bar(name='Phishing', x=["BERT"], y=[result["probabilities"]["phishing"] * 100], marker_color='red')
                    ])
                    fig.update_layout(
                        title="BERT Confidence Scores",
                        xaxis_title="Model",
                        yaxis_title="Confidence (%)",
                        barmode='group',
                        height=400
                    )
                
                return result_text, fig, "Classification successful!"
            else:
                return "Error classifying email", None, f"Error: {data.get('error', 'Unknown error')}"
        else:
            return "API Error", None, f"API Error: HTTP {response.status_code}"
    except Exception as e:
        return "Error", None, f"Error: {str(e)}"

def explain_prediction(email_text, model_name="bert"):
    try:
        response = requests.post(f"{API_URL}/explain", json={
            "text": email_text,
            "model_name": model_name
        })
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                explanation = data["explanation"]
                
                # Create text explanation
                explanation_items = explanation["explanation"]
                exp_text = f"LIME Explanation for {model_name.upper()} model:\n\n"
                
                for item in explanation_items:
                    term = item["term"]
                    weight = item["weight"]
                    direction = "INCREASES" if weight > 0 else "DECREASES"
                    exp_text += f"• '{term}' {direction} phishing probability (weight: {weight:.4f})\n"
                
                # Convert base64 image to PIL Image
                if "visualization" in explanation:
                    img_data = base64.b64decode(explanation["visualization"])
                    image = Image.open(BytesIO(img_data))
                    return exp_text, image, "Explanation generated successfully!"
                else:
                    return exp_text, None, "No visualization available"
            else:
                return "Error generating explanation", None, f"Error: {data.get('error', 'Unknown error')}"
        else:
            return "API Error", None, f"API Error: HTTP {response.status_code}"
    except Exception as e:
        return "Error", None, f"Error: {str(e)}"

def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                metrics = data["metrics"]
                
                # Create metrics summary text
                metrics_text = "Current Performance Metrics:\n\n"
                metrics_text += f"• Accuracy: {metrics['accuracy']:.2f}\n"
                metrics_text += f"• Precision: {metrics['precision']:.2f}\n"
                metrics_text += f"• Recall: {metrics['recall']:.2f}\n"
                metrics_text += f"• F1 Score: {metrics['f1']:.2f}\n"
                metrics_text += f"• AUC: {metrics['auc']:.2f}\n"
                metrics_text += f"• Evasion Rate: {metrics['evasion_rate']:.2f}\n"
                
                # Create performance evolution chart
                history = metrics["history"]
                iterations = [entry["iteration"] for entry in history]
                f1_scores = [entry["f1"] for entry in history]
                evasion_rates = [entry["evasion_rate"] for entry in history]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=iterations, y=f1_scores, mode='lines+markers', name='F1 Score'))
                fig.add_trace(go.Scatter(x=iterations, y=evasion_rates, mode='lines+markers', name='Evasion Rate'))
                fig.update_layout(
                    title="Performance Evolution",
                    xaxis_title="Iteration",
                    yaxis_title="Score",
                    height=400
                )

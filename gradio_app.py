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
                
                return metrics_text, fig, "Metrics retrieved successfully!"
            else:
                return "Error retrieving metrics", None, f"Error: {data.get('error', 'Unknown error')}"
        else:
            return "API Error", None, f"API Error: HTTP {response.status_code}"
    except Exception as e:
        return "Error", None, f"Error: {str(e)}"

def format_email_row(row):
    """Format generated email data for display"""
    result = f"**Attack Type:** {row['attack_type']}\n**Theme:** {row['theme']}\n**Model:** {row['model']}\n"
    if row.get('perturbation'):
        result += f"**Perturbation:** {row['perturbation']}\n"
    result += f"\n---\n\n{row['text']}"
    return result

def display_email_from_table(email_df, index):
    """Return the formatted email from the dataframe based on selected index"""
    if email_df is None or index < 0 or index >= len(email_df):
        return "No email selected"
    
    row = email_df.iloc[index]
    return format_email_row(row)

def sample_legitimate_email():
    """Return a sample legitimate email for demo purposes"""
    samples = [
        "Hi John, I wanted to follow up on our meeting yesterday. Can you send me the revised budget spreadsheet when you have a chance? Thanks, Sarah",
        "Dear valued customer, Thank you for your recent purchase. Your order #12345 has been shipped and should arrive in 3-5 business days. You can track your package using the following link: https://example.com/track?id=12345. Customer Service Team",
        "Hello team, This is a reminder about our weekly status meeting tomorrow at 10am. Please prepare your progress updates on current projects. Best regards, Project Management"
    ]
    return np.random.choice(samples)

# Build the Gradio UI
with gr.Blocks(title="PhishFight++: Adversarial Phishing Defense Simulator") as demo:
    gr.Markdown("""
    # PhishFight++: Adversarial Phishing Defense Simulator
    
    An interactive AI sandbox for security researchers to explore phishing email generation and detection.
    This system leverages transformer models to generate realistic phishing emails and defend against them,
    using an adversarial training approach to continuously improve detection capabilities.
    """)
    
    with gr.Tab("Attack Simulation"):
        with gr.Row():
            with gr.Column():
                attack_type = gr.Dropdown(
                    choices=config["attacker"]["attack_types"],
                    label="Attack Type",
                    value="standard"
                )
                theme = gr.Dropdown(
                    choices=config["attacker"]["themes"],
                    label="Theme",
                    value="banking"
                )
                num_samples = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Emails to Generate"
                )
                generate_btn = gr.Button("Generate Attack Emails", variant="primary")
            
            with gr.Column():
                generate_status = gr.Textbox(label="Status")
                email_table = gr.Dataframe(
                    headers=["attack_type", "theme", "model", "perturbation"],
                    label="Generated Emails",
                    interactive=True
                )
        
        email_index = gr.Number(value=0, label="Select Email Index", precision=0)
        selected_email = gr.Textbox(label="Selected Email Content", lines=10)
        
        # Connect the components
        generate_btn.click(
            generate_phishing_emails,
            inputs=[attack_type, theme, num_samples],
            outputs=[email_table, generate_status]
        )
        
        email_index.change(
            display_email_from_table,
            inputs=[email_table, email_index],
            outputs=[selected_email]
        )
    
    with gr.Tab("Defense Classification"):
        with gr.Row():
            with gr.Column(scale=1):
                input_email = gr.Textbox(
                    label="Email to Classify",
                    placeholder="Paste an email here or select from generated emails...",
                    lines=10
                )
                legitimate_sample_btn = gr.Button("Load Sample Legitimate Email")
                use_phishing_btn = gr.Button("Use Selected Phishing Email")
                use_ensemble = gr.Checkbox(label="Use Ensemble of Models", value=True)
                classify_btn = gr.Button("Classify Email", variant="primary")
            
            with gr.Column(scale=1):
                classification_result = gr.Textbox(label="Classification Result", lines=8)
                confidence_plot = gr.Plot(label="Confidence Visualization")
                classify_status = gr.Textbox(label="Status")
        
        # Connect the components
        classify_btn.click(
            classify_email,
            inputs=[input_email, use_ensemble],
            outputs=[classification_result, confidence_plot, classify_status]
        )
        
        legitimate_sample_btn.click(
            lambda: sample_legitimate_email(),
            inputs=None,
            outputs=[input_email]
        )
        
        use_phishing_btn.click(
            lambda email_text: email_text,
            inputs=[selected_email],
            outputs=[input_email]
        )
    
    with gr.Tab("Explainability"):
        with gr.Row():
            with gr.Column(scale=1):
                explain_email_input = gr.Textbox(
                    label="Email to Explain",
                    placeholder="Paste an email here or use the one from classification...",
                    lines=10
                )
                model_for_explanation = gr.Dropdown(
                    choices=["bert", "distilbert"],
                    label="Model for Explanation",
                    value="bert"
                )
                use_classified_btn = gr.Button("Use Email from Classification")
                explain_btn = gr.Button("Explain Prediction", variant="primary")
            
            with gr.Column(scale=1):
                explanation_text = gr.Textbox(label="Explanation", lines=8)
                explanation_plot = gr.Image(label="LIME Visualization")
                explain_status = gr.Textbox(label="Status")
        
        # Connect the components
        explain_btn.click(
            explain_prediction,
            inputs=[explain_email_input, model_for_explanation],
            outputs=[explanation_text, explanation_plot, explain_status]
        )
        
        use_classified_btn.click(
            lambda email_text: email_text,
            inputs=[input_email],
            outputs=[explain_email_input]
        )
    
    with gr.Tab("Evaluation Dashboard"):
        with gr.Row():
            with gr.Column():
                metrics_text = gr.Textbox(label="Performance Metrics", lines=8)
                metrics_plot = gr.Plot(label="Performance Evolution")
                metrics_status = gr.Textbox(label="Status")
                refresh_metrics_btn = gr.Button("Refresh Metrics", variant="primary")
        
        # Connect the components
        refresh_metrics_btn.click(
            get_metrics,
            inputs=None,
            outputs=[metrics_text, metrics_plot, metrics_status]
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## About PhishFight++
        
        PhishFight++ is an adversarial phishing defense system that consists of two main components:
        
        1. **PhishGen**: Generates realistic phishing emails using language models like GPT-2 and T5, with various attack types and themes.
        
        2. **PhishShield**: Defends against phishing attempts using transformer models like BERT and DistilBERT, with ensemble classification.
        
        The system uses an adversarial training approach where the defender models continuously improve based on the attacker's evolving tactics.
        
        ### Features
        
        - **Attack Simulation**: Generate realistic phishing emails with different attack patterns
        - **Defense Classification**: Classify emails as phishing or legitimate with confidence scores
        - **Explainability**: Understand why an email was classified as phishing using LIME
        - **Evaluation**: Track model performance metrics over time
        
        ### Technical Details
        
        This demo uses:
        - Transformer models for text generation and classification
        - LIME and SHAP for model explainability
        - Adversarial perturbations (synonym replacement, homoglyphs, typos)
        - Ensemble classification for improved detection
        
        ### Note
        
        This is a demonstration system. For real-world deployment, consider:
        - Training on larger, more diverse datasets
        - Regular model updates and fine-tuning
        - Integration with email security infrastructure
        - Human-in-the-loop review processes
        """)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)

# DeepMental: Leveraging Fine-Tuning Reasoning Large Language Models (LLMs) to Enhance Mental Health Support Chatbots

## Project Overview

This project evaluates the effectiveness of a reasoning-based LLM (DeepSeek Llama 8B) for delivering structured, empathetic, and contextually appropriate responses in mental health counselling. Three approaches are compared:
- A reasoning-based fine-tuned model (DeepMental)
- A traditional fine-tuned model
- A few-shot learning baseline

## Live Demo

The DeepMental model is deployed and accessible through Hugging Face Spaces. The deployment uses Gradio for the user interface and supports real-time conversation with the model.

## Technical Requirements

The project uses Python with the following key dependencies:
- Hugging Face Transformers
- Unsloth for model optimization
- Gradio for the web interface
- PyTorch for deep learning
- LangChain for conversation memory management

## Repository Structure

### Data Processing and Analysis
1. `0. Public_Real_Dataset_Processing_.ipynb`
   - Processes and prepares public mental health conversation datasets

2. `1. EDA_of_Synthetic_Data.ipynb`
   - Exploratory Data Analysis of the synthetic CBT dataset
   - Includes demographic analysis, diversity measures, correlation matrices
   - Term frequency analysis across therapy phases

### Model Training
3. `2. Few Shot Learning.ipynb`
   - Implements few-shot learning baseline using 100 annotated examples
   - Uses QLoRA for efficient training

4. `3. Fine Tune - DeepMental.ipynb`
   - Full fine-tuning of the DeepMental model using QLoRA
   - Incorporates THINK methodology for step-by-step reasoning
   - Optimizes for CBT alignment

### Response Generation
5. `4. Generate-traditional.ipynb`
   - Generates responses using traditionally fine-tuned LLaMA model
   - Uses Alpaca-style prompt formatting

6. `5. Generate-fewshot.ipynb`
   - Implements few-shot learning response generation
   - Maintains consistent prompt structure for comparison

7. `6. Generate Responses - DeepMental.ipynb`
   - Generates responses using the DeepMental reasoning model
   - Produces structured outputs with cognitive restructuring patterns

### Evaluation and Analysis
8. `7. gather generated data.ipynb`
   - Processes and merges outputs from all models
   - Standardizes format and aligns with benchmark dataset

9. `8. Inference - DeepMental.ipynb`
   - Individual inference testing
   - Supports custom input scenarios

10. `9. Evalution.ipynb`
    - Comprehensive evaluation using MentalChat16K benchmark
    - Metrics: ROUGE, BLEU, BERTScore, Distinct-n, Perplexity

11. `10. Evalution Visuliztion.ipynb`
    - Visualization of evaluation metrics
    - Comparative analysis across models

### Deployment
12. `11. Save and Upload to huggingface.ipynb`
    - Model saving and Hugging Face upload procedures

13. `12. app on huggingface.py`
    - Gradio-based web interface implementation
    - Real-time conversation handling
    - Memory management for context retention

## Generated Data
The repository includes generated responses from all three models:
- `generated_Fine_Tune_DeepMental_responses.csv`
- `generated_Fine_Tune_Traditional_responses.csv`
- `generated_Few_Shot_responses.csv`

## Model Architecture

The DeepMental model is built on the DeepSeek Llama 8B architecture and incorporates:
- QLoRA for efficient fine-tuning
- Conversation memory management
- Structured reasoning patterns (THINK methodology)
- Real-time response generation

## Deployment Details

The model is deployed using:
- Gradio for the web interface
- Hugging Face Spaces for hosting
- Conversation buffer memory for context retention
- Streamed text generation for responsive user experience

Settings for inference:
- Max tokens: 510
- Temperature: 0.7
- Top-p: 0.95
- Top-k: 50
from unsloth import FastLanguageModel
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
import torch
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ibrahimShaban/Fine-Tuned-Deepmental-Llama-8b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


# Memory for storing conversation history
memory = ConversationBufferMemory()

# Alpaca prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n"""

def generate_response(query, input_context="", max_new_tokens=510):
    prompt = ALPACA_PROMPT.format(instruction=query, input=input_context)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
    return generated_text

def chat(query, history):
    history = history or []
    memory.save_context({"input": query}, {"output": "Generating response..."})
    response = generate_response(query)
    memory.save_context({"input": query}, {"output": response})
    history.append((query, response))
    return history, ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Enter your query", placeholder="Type your message here...")
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear Conversation")

    submit_button.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_button.click(lambda: ([], ""), outputs=[chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True, inbrowser=True)
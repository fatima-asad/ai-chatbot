import gradio as gr
from transformers import pipeline
from PIL import Image

# 1. Setup the Vision Model
pipe = pipeline("image-to-text", model="HuggingFaceTB/SmolVLM-Instruct")

def chat_and_detect(message, history):
    user_text = message["text"].lower()
    
    # KNOWLEDGE BASE: PAKISTAN 2026
    pakistan_info = {
        "president": "As of 2026, the President of Pakistan is **Asif Ali Zardari**.",
        "prime minister": "The Prime Minister of Pakistan is **Shehbaz Sharif**.",
        "capital": "The capital of Pakistan is **Islamabad**.",
    }

    # Check for facts
    for key in pakistan_info:
        if key in user_text:
            return pakistan_info[key]

    # IMAGE DETECTION
    if message["files"]:
        image_path = message["files"][0]
        img = Image.open(image_path)
        prompt = f"User: <image>\nDescribe this image.\nAssistant:"
        result = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 100})
        description = result[0]["generated_text"].split("Assistant:")[-1].strip()
        return f"📸 **Detection:** {description}"
    
    return "I am ready! Ask about Pakistan or upload an image."

# --- CUSTOM BEAUTIFUL UX/UI CSS ---
custom_css = """
/* The Main App Background */
.gradio-container {
    background-color: #f0f2f5 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* BOT MESSAGE: Blue Background + White Text */
.message.bot {
    background-color: #007bff !important; /* Professional Blue */
    color: white !important;               /* Pure White Text */
    border-radius: 15px 15px 15px 2px !important;
    border: none !important;
    font-size: 16px !important;
}

/* USER MESSAGE: Darker Blue or Gray for contrast */
.message.user {
    background-color: #e4e6eb !important;
    color: #050505 !important;
    border-radius: 15px 15px 2px 15px !important;
}

/* Make the title look like a real website header */
#custom-title {
    text-align: center;
    color: #007bff;
    font-weight: bold;
    padding-bottom: 20px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<h1 id='custom-title'>🇵🇰 Pakistan Vision AI Portal</h1>")
    
    chat = gr.ChatInterface(
        fn=chat_and_detect,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            placeholder="Type here or upload an image...",
            file_types=["image"]
        )
    )

if __name__ == "__main__":
    demo.launch()
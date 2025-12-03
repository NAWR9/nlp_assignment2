import json
from pathlib import Path

import gradio as gr

from run_text_generation import (
    load_config,
    load_tokenizer,
    load_text_model,
    generate_text,
)


BASE_DIR = Path(__file__).resolve().parent


# Load model and tokenizer once at startup
config = load_config()
tokenizer = load_tokenizer()
model = load_text_model()
max_sequence_len = config.get("max_sequence_len", 40)


def generate_arabic(seed_text: str, num_words: int, temperature: float):
    if not seed_text.strip():
        return ""

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        seed_text=seed_text.strip(),
        num_words=num_words,
        max_sequence_len=max_sequence_len,
        temperature=temperature,
    )
    return text


with gr.Blocks(css="""
body { direction: rtl; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.gradio-container { max-width: 800px; margin: auto; }
""") as demo:
    gr.Markdown("## مولد نصوص عربية (نموذج LSTM محفوظ محلياً)")
    gr.Markdown(
        "اكتب بداية الجملة بالعربية، ثم اختر عدد الكلمات ودرجة الحرارة، "
        "واضغط على **توليد النص** لعرض النتيجة."
    )

    with gr.Row():
        with gr.Column():
            seed_input = gr.Textbox(
                label="نص البداية (بالعربية)",
                placeholder="مثال: هذا الكتاب",
            )
            num_words_slider = gr.Slider(
                label="عدد الكلمات المولدة",
                minimum=5,
                maximum=100,
                step=5,
                value=20,
            )
            temperature_slider = gr.Slider(
                label="درجة الحرارة (0 = اختيار حتمي، القيم الأعلى = أكثر عشوائية)",
                minimum=0.0,
                maximum=1.5,
                step=0.1,
                value=1.0,
            )
            generate_button = gr.Button("توليد النص")

        with gr.Column():
            output_box = gr.Textbox(
                label="النص الناتج",
                lines=8,
            )

    generate_button.click(
        fn=generate_arabic,
        inputs=[seed_input, num_words_slider, temperature_slider],
        outputs=output_box,
    )


if __name__ == "__main__":
    # Launch Gradio UI on localhost so the browser can open it directly
    demo.launch(server_name="127.0.0.1", server_port=7860)



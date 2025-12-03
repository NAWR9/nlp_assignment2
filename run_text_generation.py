import json
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models" / "text_generation"


def load_config():
    config_path = MODEL_DIR / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer():
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    # `tokenizer_from_json` in tf.keras expects a JSON string
    with tokenizer_path.open("r", encoding="utf-8") as f:
        json_string = f.read()
    return tokenizer_from_json(json_string)


def load_text_model():
    model_path = MODEL_DIR / "lstm_text_generator.keras"
    return load_model(model_path)


def generate_text(model, tokenizer, seed_text, num_words, max_sequence_len, temperature: float = 1.0):
    """
    Generate text by iteratively predicting the next word.
    """
    result = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre",
        )

        # Predict probabilities for next word
        preds = model.predict(token_list, verbose=0)[0]

        # Optional temperature sampling
        if temperature is not None and temperature > 0:
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds + 1e-8) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            next_index = np.random.choice(len(preds), p=preds)
        else:
            next_index = int(np.argmax(preds))

        # Map index back to word
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                result += " " + word
                break
        else:
            # If index not found (rare), stop generation
            break

    return result


def main():
    config = load_config()
    tokenizer = load_tokenizer()
    model = load_text_model()

    max_sequence_len = config.get("max_sequence_len", 40)

    print("Loaded text generation model.")
    print("Type a seed phrase and press Enter to generate text.")
    print("Press Enter on an empty line to quit.\n")

    # Some example prompts for quick testing (Arabic)
    example_prompts = [
        "القصة كانت",
        "أعتقد أن الكاتب",
        "هذا الكتاب",
    ]

    print("أمثلة يمكن تجربتها:")
    for p in example_prompts:
        print(f"  - {p}")
    print()

    while True:
        seed_text = input("Seed text (empty to quit): ").strip()
        if not seed_text:
            break

        try:
            num_words_str = input("How many words to generate [default 20]: ").strip()
            num_words = int(num_words_str) if num_words_str else 20
        except ValueError:
            num_words = 20

        try:
            temp_str = input("Temperature (0 = greedy, typical 0.7–1.0) [default 1.0]: ").strip()
            temperature = float(temp_str) if temp_str else 1.0
        except ValueError:
            temperature = 1.0

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed_text,
            num_words=num_words,
            max_sequence_len=max_sequence_len,
            temperature=temperature,
        )
        print("\nGenerated text:")
        print(generated)
        print("-" * 80)


if __name__ == "__main__":
    main()



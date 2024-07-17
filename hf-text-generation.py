import transformers
from transformers import pipeline


def main():
    text_generator = pipeline("text-generation", model="gpt2")
    # Set seed for reproducibility
    transformers.set_seed(1)
    prompt = "AI is the future of technology because"
    synthetic_text = text_generator(prompt, num_return_sequences=5, max_new_tokens=50)
    for text in synthetic_text:
        print(text.get("generated_text"), "\n---------------")


if __name__ == "__main__":
    main()

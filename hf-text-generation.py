from transformers import pipeline


def main():
    text_generator = pipeline("text-generation", model="gpt2")
    prompt = "AI is the future of technology because"

    synthetic_text = text_generator(prompt, num_return_sequences=5, # Generate 5 different completions
                                    max_new_tokens=500, # Generate up to 500 tokens
                                    temperature=0.2, # Low temperature for more deterministic results
                                    repetition_penalty=1.2, # Penalize repeated text
                                    )

    for text in synthetic_text:
        print(text.get("generated_text"), "\n---------------")


if __name__ == "__main__":
    main()

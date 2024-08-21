from transformers import pipeline

def main() :
    summarizer = pipeline("summarization", model=model)

    text = """This is a long text that we want to summarize. 
    It can be about any topic, such as the environment, technology, or politics. 
    We'll provide a detailed description of the subject matter, including relevant facts, figures, and expert opinions. 
    The goal is to create a comprehensive overview that captures the essence of the original text.
    """

    summary = summarizer(text)

    print("Text: ", text)
    print("Summary: ", summary[0]['summary_text'])

if __name__ == "__main__":
    model = "google-t5/t5-small"

    main()
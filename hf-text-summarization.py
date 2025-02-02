from transformers import pipeline
import truecase
import nltk


def check_and_download_nltk_data(dataset_name):
    """
    Check if the NLTK dataset is downloaded and download it if it is not
    :param dataset_name: The name of the dataset
    :return: None
    """
    try:
        # Try to use the dataset
        nltk.data.find(f'tokenizers/{dataset_name}')
        print(f"The '{dataset_name}' dataset is already downloaded.")
    except LookupError:
        # If the dataset is not found, download it
        print(f"The '{dataset_name}' dataset is not found. Downloading it now...")
        nltk.download(dataset_name)

def main() :

    check_and_download_nltk_data('punkt_tab')

    summarizer = pipeline("summarization", model=model)

    text = """This is a long text that we want to summarize. 
    It can be about any topic, such as the environment, technology, or politics. 
    We'll provide a detailed description of the subject matter, including relevant facts, figures, and expert opinions. 
    The goal is to create a comprehensive overview that captures the essence of the original text.
    """

    summary = summarizer(text)

    print("Text: ", text)
    print("Summary: ", truecase.get_true_case(summary[0]['summary_text']))

if __name__ == "__main__":
    model = "google-t5/t5-small"

    main()
from datasets import load_dataset_builder
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

"""
This script demonstrates how to load and inspect a dataset using the Hugging Face datasets library
"""

_dataset_name = 'rotten_tomatoes'
_model_name = 'bert-base-uncased'


def inspect():
    """
    Inspect the dataset and print the info without downloading the dataset
    :return: None
    """

    ds_builder = load_dataset_builder(_dataset_name)

    print("Info: ", ds_builder.info)
    print("Description: ", ds_builder.info.description)
    print("Features: ", ds_builder.info.features)


def load():
    """
    Download the load and download dataset
    :return: None
    """

    dataset = load_dataset(_dataset_name, split='train')
    print(dataset[0]['text'])


def preprocess():
    """
    Preprocess the dataset
    :return: None
    """
    tokenizer = AutoTokenizer.from_pretrained(_model_name)
    dataset = load_dataset(_dataset_name, split="train")
    # Save the dataset to a json file.
    # dataset.to_json(f"./tmp/dataset/{_dataset_name}.json")

    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
    print("Tokenized: ", tokenized_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["label"],
        batch_size=2,
        collate_fn=data_collator,
        shuffle=True
    )

    print("Tensorflow Dataset: ", tf_dataset)


if __name__ == "__main__":
    # inspect()
    # load()
    preprocess()

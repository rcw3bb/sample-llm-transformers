import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer
from transformers import TFAutoModelForSequenceClassification

model_name = 'distilbert-base-uncased'


def tokenize(batch):
    db_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return db_tokenizer(batch['verse_text'], padding=True, truncation=True)


def create_model(enc_poem_sentiment):
    print("Tensorflow version:", tf.__version__)
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    training_dataset = enc_poem_sentiment['train']
    labels = training_dataset.features.get('label')
    num_labels = len(labels.names)
    sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    sentiment_model.get_config()

    print(sentiment_model)
    sentiment_model.layers[0].trainable = True

    sentiment_model.summary()

    return sentiment_model


def train_model(enc_poem_sentiment):
    db_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    training_dataset = enc_poem_sentiment['train']
    validation_dataset = enc_poem_sentiment['validation']
    batch_size = 64
    labels = training_dataset.features.get('label')
    num_labels = len(labels.names)
    tokenizer_columns = db_tokenizer.model_input_names

    # Convert dataset to tensorflow
    training_dataset = training_dataset.to_tf_dataset(columns=tokenizer_columns, label_cols='label', shuffle=True,
                                   batch_size=batch_size)
    validation_dataset = validation_dataset.to_tf_dataset(columns=tokenizer_columns, label_cols='label', shuffle=False,
                                     batch_size=batch_size)
    sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Define optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

    sentiment_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Fine-tune the model
    sentiment_model.fit(training_dataset, validation_data=validation_dataset, epochs=5)
    return sentiment_model


def predict_sentiment(enc_poem_sentiment, sentiment_model):
    db_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    training_dataset = enc_poem_sentiment['train']
    labels = training_dataset.features.get('label')
    batch_size = 64
    tokenizer_columns = db_tokenizer.model_input_names
    infer_data = {'id': [0, 1],
                  'verse_text': [
                      "and be glad in the summer morning",
                      "how hearts were answering to his own"
                  ],
                  'label': [1, 0]}
    infer_dataset = Dataset.from_dict(infer_data)
    ds_dict = DatasetDict()
    ds_dict['infer'] = infer_dataset

    enc_dataset = ds_dict.map(tokenize, batched=True, batch_size=None)
    infer_final_dataset = enc_dataset['infer'].to_tf_dataset(columns=tokenizer_columns, shuffle=True,
                                                             batch_size=batch_size)
    predictions = sentiment_model.predict(infer_final_dataset)
    print("Logits", predictions.logits)
    pred_lable_ids = np.argmax(predictions.logits, axis=1)
    for i in range(len(pred_lable_ids)):
        print("Poem = ", infer_data['verse_text'][i],
              " Predicted = ", labels.names[pred_lable_ids[i]],
              " True-Label = ", labels.names[infer_data['label'][i]])


def main():
    # The hugging face dataset.
    dataset_name = 'poem_sentiment'

    poem_sentiment = load_dataset(dataset_name)

    print("Tensorflow Version:", tf.__version__)
    print("CPU:", tf.reduce_sum(tf.random.normal([1000, 1000])))
    print("GPU:", tf.config.list_physical_devices('GPU'))

    print(poem_sentiment)
    print(poem_sentiment['test'][20:25])

    print("\nSentiment labels used:", poem_sentiment['train'].features['label'].names)

    enc_poem_sentiment = poem_sentiment.map(tokenize, batched=True, batch_size=None)

    print(enc_poem_sentiment['train'][0:5])

    # create_model(enc_poem_sentiment)
    model = train_model(enc_poem_sentiment)
    predict_sentiment(enc_poem_sentiment, model)


if __name__ == "__main__":
    main()

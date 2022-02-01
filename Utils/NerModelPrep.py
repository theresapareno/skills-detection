import json
from tqdm import tqdm
from spacy.tokens import DocBin
import srsly
import spacy
from sklearn.model_selection import train_test_split
from pathlib import Path

# https://spacy.io/usage/training


nlp = spacy.blank("en")


# get the ubiai data
def load_data(data_path: Path):
    with open(data_path, 'rb') as jf:
        data = json.load(jf)
    return data


def convert_json(data):
    bigList = []
    for i in data:
        try:
            sentence = i['document']
            entities = i['annotation']
            entityList = []
            if len(entities) > 1:
                for j in entities:
                    start = j['start']
                    end = j['end']
                    label = j['label']
                    triple = (start, end, str(label))
                    entityList.append(triple)
            else:
                start = entities['start']
                end = entities['end']
                label = entities['label']
                triple = (start, end, str(label))
                entityList.append(triple)
        except Exception as e:
            print(e)
            print(i)
        bigList.append((sentence, {'entities': entityList}))

    return bigList


# convert json to .spacy file
def create_training(TRAIN_DATA):
    db = DocBin()
    for text, annot in tqdm(TRAIN_DATA):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db


def get_raw_text(path: Path, data):
    raw_text = []
    for i in data:
        sentences = i["document"]
        sent = {"text": sentences}
        raw_text.append(sent)
    srsly.write_json(path + "\\raw_text.jsonl", data)
    print("Successfully stored the raw text that will be used for Bi-LSTM! ")


def create_split_data(path, data, test_size=0.2):
    train_data, valid_data = train_test_split(data, test_size=test_size)
    print("Length of training data: ", len(train_data))
    print("Length of valid data: ", len(valid_data))
    train_data = create_training(train_data)
    train_data.to_disk(path + "\\train_data.spacy")
    valid_data = create_training(valid_data)
    valid_data.to_disk(path + "\\valid_data.spacy")
    print("Successfully created training and valid data")


def prepareData(path, training_path):
    """
    This will convert the json from ubiai to .spacy format.
    :param path:
    :param training_path:
    :return:
    """

    # gets the original ubiai output
    data = load_data(path)

    big_list = convert_json(data)
    get_raw_text(training_path, data)
    create_split_data(training_path, big_list)
    print("Successfully created all necessary training data!")


training_path = r"../Data/NER_trainingData"

path = r"../Data/UBIAI/output/ner.json"
prepareData(path, training_path)

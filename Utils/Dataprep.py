import os
import re
from html.parser import HTMLParser
from io import StringIO
from Utils.GenerateData import generate_entities
import en_core_web_sm
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import srsly
import warnings
import json
warnings.filterwarnings("ignore")


def create_lang_detector(nlp, name):
    return LanguageDetector()

Language.factory("language_detector", func=create_lang_detector)
nlp = en_core_web_sm.load(disable=["tagger", "ner"])
nlp.max_length = 2000000
nlp.add_pipe('language_detector', last=True)

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def load_file(path, filename):
    file = os.path.join(path, filename)
    f = open(file, encoding="utf8")
    text = f.readlines()
    return text

def load_data(data_path): # get the ubiai data
    with open(data_path, 'rb') as jf:
        data = json.load(jf)
    return data

def get_title(file):
    return file[0]


def get_location(file):
    return file[2]


def get_company(file):
    return file[1]


def spacy_get_language(file):
    string = " ".join(file)
    doc = nlp(string)
    return doc._.language['language']


def extract_li(file):
    no_lb = str(file).replace('\n', '')
    p = re.findall(r'<li>(.*?)</li>', str(no_lb))
    if len(p) > 0:
        return p[0]
    else:
        return file[0]

def strip_tags(file):
    s = MLStripper()
    if len(file) > 0:
        sentences = []
        for line in range(len(file)):
            sent = file[line]
            s.feed(sent)
            if len(s.get_data()) > 1:
                sentences.append(clean_content(s.get_data()))
                s = MLStripper()
            else:
                continue
                s = MLStripper()
        return sentences
    else:
        pass

def clean_content(content):
    pattern = "\\n|<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"
    compiled_pat = re.compile(pattern)
    cleantext = re.sub(compiled_pat, '', content)
    return cleantext

def file_combined(file):
    return " ".join(file)


def get_more_data(path,ubi_path,training_path):
    new_data = []
    for filename in os.listdir(path):
        file = load_file(path, filename)
        file = strip_tags(file)
        file_en = spacy_get_language(file)
        # if english or german apply 'en' model or 'de' model.
        if file_en == "en":
            title = get_title(file)  # insert this to the db
            location = get_location(file)  # insert this to the db
            company = get_company(file)  # insert this to the db
            file_en # insert this to the db
        else:
            continue
        all_text = file_combined(file)
        job_training = generate_entities(all_text) # insert this to the db
        new_data.append(job_training[0])

    # combine old labelled data to new job entities.
    ubiai_ner_data = load_data(ubi_path) # load ubiai and combine new_data
    merged_data = ubiai_ner_data + new_data
    print(len(merged_data))
    #srsly.write_json(training_path + "\\all_training.json", merged_data)
    print("Successfully Stored all new dataset in NER Training folder!")




path = r"C:\Users\there\PycharmProjects\careerguide\Data\Raw data"
ubi_data_path = r"C:\Users\there\PycharmProjects\careerguide\Data\UBIAI\output\ner.json"
training_path = r"C:\Users\there\PycharmProjects\careerguide\Data\NER_trainingData"
get_more_data(path,ubi_data_path,training_path)



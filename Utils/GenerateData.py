import spacy
import re


# load model.
ner_model = r"C:\Users\there\PycharmProjects\careerguide\Models\NERModel\output_128\model-best"
nlp = spacy.load(ner_model)

def get_unique_tokens(annotation):
    unique = []
    all_annot = []
    for i in annotation:
        start = i['start']
        end = i['end']
        if start in unique and end in unique:
            continue
        else:
            unique.append(start)
            unique.append(end)
            all_annot.append(i)
    return all_annot


def generate_entities(text):
    """
    This gets the entities of the job description.
    :param text:
    :param new_program:
    :return: combine with old ubiai old data.
    """
    Job = []
    annotations = []
    new_skills = ["f#","julia","elm","ruby","swift","kotlin"]
    small_text = text.lower()
    doc = nlp(text)
    for i in new_skills:
        pat = f"(?<![\w\d]){i}(?![\w\d])"
        for word in re.finditer(pat,small_text):
            # print("difference: ",i, len(i),(word.start()+len(i) - word.start()))
            if len(i) == (word.start()+len(i) - word.start()):
                # print(word.start(), word.start()+len(i)) # position
                annot = {"start":word.start(),"end":word.start()+len(i),
                         "label":"SKILLS","text":i}
                annotations.append(annot)
    for ent in doc.ents:
        #print(ent.text +" : "+ ent.label_ , ent.start_char , ent.end_char+1)
        annot_spacy = {"start": ent.start_char, "end": ent.end_char,
                 "label": ent.label_, "text": ent.text}
        annotations.append(annot_spacy)

    newAnnot = sorted(get_unique_tokens(annotations), key=lambda d: d['start'])
    jobLabel = {"document":text,"annotation":newAnnot,"user_input": ""}
    Job.append(jobLabel)

    return Job


def generate_relations(text):
    pass


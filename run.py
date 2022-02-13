import spacy

# insert job description
job_description = """
Master or Ph.D. in Mathematics, Physics, Computer Science or a similar field
Three to six years of professional experience outside academia, preferably experience in Supply Chain
Strong skills in Machine Learning, Statistics and Predictive Modelling
Project management experience
Programming skills, e.g. Python, R, Java, C++
Comfortable reporting results to both technical and non-technical audiences
Ability to work with people from various backgrounds/cultures
Strong interpersonal skills with the desire to share knowledge and skills
Preferably experience in Visualization Tools, e.g. Tableau, Qlik as well as Data processing and/or database programming skills, e.g. SQL 
"""


ner_model_path = r"Models/model-best"

ner_model = spacy.load(ner_model_path)

# test the algorithm
doc = ner_model(job_description)

for ent in doc.ents:
    print(ent.text, '--->', ent.label_)

from allennlp.predictors.predictor import Predictor
import json

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
# predictor.predict(document="The woman reading a newspaper sat on the bench with her dog.")

with open('coqa-dev-v1.0.json', 'r') as f:
    dataset = json.load(f)
data = dataset['data']
for article in data:
    text = article['story']
    questions = article['questions']
    answers = article['answers']
    for (question, answer) in zip(questions, answers):
        q_text = question['input_text']
        a_text = answer['input_text']
        text = text + " # " + q_text + " # " + a_text
    print(text)
    # prediction = predictor.predict(document=text)
    # print(prediction)
    break
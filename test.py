import json
import spacy

if __name__ == '__main__':
    with open('dataset/coqa/coqa-dev-v1.0.json', 'r') as f:
        data = json.load(f)['data']
    article = data[0]
    text = article['story']
    for (question, answer) in zip(article['questions'], article['answers']):
        text += question['input_text'] + ' '
        text += answer['input_text'] + ' '
    nlp = spacy.load('en_coref_sm')
    doc = nlp(text)
    print(doc._.has_coref)
    print(doc._.coref_clusters)  # 共指链组成的列表
    print(doc._.coref_resolved)
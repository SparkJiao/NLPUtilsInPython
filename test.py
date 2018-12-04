import json
import spacy
import re

if __name__ == '__main__':
    # with open('dataset/coqa/coqa-dev-v1.0.json', 'r') as f:
    #     data = json.load(f)['data']
    # article = data[0]
    # text = article['story']
    # for (question, answer) in zip(article['questions'], article['answers']):
    #     text += question['input_text'] + ' '
    #     text += answer['input_text'] + ' '
    # nlp = spacy.load('en_coref_sm')
    # doc = nlp(text)
    # print(doc._.has_coref)
    # print(doc._.coref_clusters)  # 共指链组成的列表
    # print(doc._.coref_resolved)
    a = 'she is a beautiful girl. I love her. And he is my bro. he!'
    pattern = re.compile(r'[^a-zA-Z]he[^a-zA-Z]')
    s = pattern.sub(' [mask] ', a)
    print(s)
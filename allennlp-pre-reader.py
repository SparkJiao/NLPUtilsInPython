import argparse
import json
import logging
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField
from allennlp.data.vocabulary import Vocabulary
import torch

import bert_indexer

logger = logging.getLogger()
#
# parser = argparse.ArgumentParser('description: experiments on datasets')
# parser.add_argument('input_file')
# parser.add_argument('output_file')
# args = parser.parse_args()

tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True, ner=True))
token_indexer = bert_indexer.PretrainedBertIndexer('../TransformerCoqa/bert-base-uncased-vocab.txt', do_lowercase=False, max_pieces=8, doc_stride=3)
token_embedder = PretrainedBertEmbedder('../TransformerCoqa/bert-base-uncased.tar.gz')

# with open(args.input_file, 'w') as f:
#     data = json.load(f)['data']
#
# for article in data:
#     story = article['story']

a = "the man went to the store and bought a gallon of milk"
b = tokenizer.tokenize(a)
print(b)

bert_vocab = Vocabulary()
c = token_indexer.tokens_to_indices(b, bert_vocab, 'bert')
print(c)

input_ids = c['bert']
for input_id in input_ids:
    tokens = [bert_vocab.get_token_from_index(index=idx, namespace='bert') for idx in input_id]
    print(tokens)

d = token_embedder(torch.LongTensor(c['bert']))
print(d.size())

e = token_embedder(torch.LongTensor(c['bert']), torch.LongTensor(c['bert-offsets']))
print(e.size())

# d = TextField(b, {'bert': token_indexer})
# print(b)
#
# sentence1 = a
# sentence2 = "Fangkai Jiao is under guided of Dr.Huang and Dr.Nie since 2018.9"
#
# tokens1 = tokenizer.tokenize(sentence1)
# tokens2 = tokenizer.tokenize(sentence2)
# print("tokens1: ")
# print(tokens1)
# print("tokens2: ")
# print(tokens2)
#
# vocab = Vocabulary()
#
# instance1 = Instance({"passage": TextField(tokens1, {"bert": token_indexer})})
# instance2 = Instance({"passage": TextField(tokens2, {"bert": token_indexer})})
#
# batch = Batch([instance1, instance2])
# batch.index_instances(vocab)
#
# padding_lengths = batch.get_padding_lengths()
# print("Padding length: ", padding_lengths)
# tensor_dict = batch.as_tensor_dict(padding_lengths)
# tokens = tensor_dict["passage"]
#
# # 16 = [CLS], 17 = [SEP]
# print(tokens['bert'])
#
# print(tokens['bert-offsets'])
#
# # No offsets, should get 14 vectors back ([CLS] + 12 token wordpieces + [SEP])
# bert_vectors = token_embedder(tokens["bert"])
# print(bert_vectors.size())
#
# # Offsets, should get 10 vectors back.
# bert_vectors = token_embedder(tokens["bert"], offsets=tokens["bert-offsets"])
# print(bert_vectors.size())
#
# # Now try top_layer_only = True
# tlo_embedder = PretrainedBertEmbedder('../TransformerCoqa/bert-base-uncased.tar.gz', top_layer_only=True)
# bert_vectors = tlo_embedder(tokens["bert"])
# print(bert_vectors.size())
#
# bert_vectors = tlo_embedder(tokens["bert"], offsets=tokens["bert-offsets"])
# print(bert_vectors.size())

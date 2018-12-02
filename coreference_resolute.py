from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
predictor.predict(document="The woman reading a newspaper sat on the bench with her dog.")

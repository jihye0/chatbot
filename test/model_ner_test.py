import sys
sys.path.append('/Users/jihye/FLYAI/chatbot')
from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='/Users/jihye/FLYAI/chatbot/chatbot_dict.bin',
               userdic='/Users/jihye/FLYAI/chatbot/utils/user_dic.tsv')


ner = NerModel(model_name='/Users/jihye/FLYAI/chatbot/ner_model.h5', proprocess=p)
query = '오늘 오전 13시 2분에 탕수육 주문 하고 싶어요'
predicts = ner.predict(query)
tags = ner.predict_tags(query)
print(predicts)
print(tags)
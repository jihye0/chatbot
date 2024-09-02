import sys
sys.path.append('/Users/jihye/FLYAI/chatbot')
from utils.Preprocess import Preprocess
from models.intent.intentModel import IntentModel

p = Preprocess(word2index_dic='/Users/jihye/FLYAI/chatbot/chatbot_dict.bin',
               userdic='/Users/jihye/FLYAI/chatbot/utils/user_dic.tsv')

intent = IntentModel(model_name='/Users/jihye/FLYAI/chatbot/intent_model.h5', proprocess=p)
query = "씨벌 전화좀 받아라"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)
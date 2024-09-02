# 필요한 모듈 임포트
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

# 데이터 읽어오기
train_file = "/Users/jihye/FLYAI/chatbot/models/intent/total_train_data.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data['query'].tolist()
intents = data['intent'].tolist()

import sys
sys.path.append('/Users/jihye/FLYAI/chatbot')
from utils.Preprocess import Preprocess
p = Preprocess(word2index_dic='/Users/jihye/FLYAI/chatbot/chatbot_dict.bin',
               userdic='/Users/jihye/FLYAI/chatbot/utils/user_dic.tsv')

# 단어 시퀀스 생성
sequences = []
for sentence in queries:
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag=True)
    seq = p.get_wordidx_sequence(keywords)
    sequences.append(seq)

# 위에서 생성한 단어 시퀀스 벡터의 크기를 동일하게 맞춰주기 위해 MAX_SEQ_LEN 크기만큼 시퀀스 벡터를 패딩 처리합니다.
# 단어 인덱스 시퀀스 벡터 
# 단어 시퀀스 벡터 크기
from GlobalParams import MAX_SEQ_LEN
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

# (105658, 15)
print(padded_seqs.shape)
print(len(intents)) #105658

# 학습용, 검증용, 테스트용 데이터셋 생성 
# 학습셋:검증셋:테스트셋 = 7:2:1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

# 케라스 함수형 모델 방식으로 의도 분류 모델을 구현합니다. 입력하는 문장을 의도 클래스로 분류하는 CNN모델은 여러 영역으로 구성되어 있습니다. 첫 번째로 입력 데이터를 단어 임베딩 처리하는 영역, 그 다음으로 합성곱 필터와 연산을 통해 특징맵을 추출하고 평탄화하는 영역, 완전 연결 계층을 통해 감정별로 클래스를 분류하는 영역으로 구성되어 있습니다.
# 하이퍼 파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1 #전체 단어 개수

# CNN 모델 정의 
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3,4,5gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

# 우리는 5가지 의도 클래스를 분류해야하기에 출력노드가 5개인 Dense 계층을 생성합니다. 마지막으로 출력 노드로 정의한 logits에서 나온 함수를 소프트맥스 계층을 통해 감정 클래스별 확률을 계산합니다.
hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(5, name='logits')(dropout_hidden)
predictions = Dense(5, activation=tf.nn.softmax)(logits)

# 위에서 정의한 계층들을 케라스 모델에 추가하는 작업을 진행합니다. 모델 정의 후 실제 모델을 model.compile() 함수를 통해 CNN모델을 컴파일합니다. 최적화 방법에는 adam, 손실함수에는 sparse_categorical_crossentropy, 모델 평가할 때 정확도 확인하기 위해 metrics에 accuracy를 사용하도록 했습니다.
# 모델 생성 
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델을 학습하기 위해 model.fit() 함수를 사용합니다. 에포크값을 5로 설정했으므로 모델 학습을 5회 반복합니다. 또한 evaluate() 함수를 이용해 성능을 평가합니다. 인자에는 테스트용 데이터셋을 사용합니다. 마지막으로 학습이 완료된 모델을 h5 파일 포맷으로 저장합니다. 해당 모델 파일은 챗봇 엔진의 의도 분류 모델에서 사용됩니다.
# 모델 학습 
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# 모델 평가(테스트 데이터 셋 이용) 
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
print('loss: %f' % (loss))

# 모델 저장  
model.save('intent_model.h5')
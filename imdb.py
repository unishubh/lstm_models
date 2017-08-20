from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
data = pd.read_csv('imdb/train.tsv',  sep='\t')
data = data[['Phrase','Sentiment']]
data['Phrase'] = data['Phrase'].apply(lambda x:x.lower())
max_features = 3500
tokenizer = Tokenizer(num_words = max_features, split=' ')
tokenizer.fit_on_texts(data['Phrase'].values)
X = tokenizer.texts_to_sequences(data['Phrase'].values)
X = pad_sequences(X)

embed = 128
lstm = 196
model = Sequential()
model.add(Embedding(max_features,embed,input_length = X.shape[1],dropout=0.3))
model.add(LSTM(lstm,dropout_W=0.3,dropout_U=0.3))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print model.summary()
Y = pd.get_dummies(data['Sentiment'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 600
X_train = np.array(X_train)
Y_train = np.array(Y_train)
model.fit(X_train, Y_train, epochs = 40, batch_size=batch_size, verbose = 2)
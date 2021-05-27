#%%

import pandas as pd
import numpy as np

## initial trial with 500 normal and abnormal transcripts

#allnormal=np.load('allnormal30875_first500.npy', allow_pickle=True)
#allabnormal=np.load('allabnormal30875_first500.npy', allow_pickle=True)


# Load full SAD Human Body Map ERR030875 data
allnormal=np.load("/home/priyamvada/data/allnormal30875.npy",allow_pickle=True)
allabnormal=np.load("/home/priyamvada/data/allabnormal30875.npy",allow_pickle=True)

column_names=('transcript_name', 'expected_coverage', 'observed_coverage', 'label')
df_normal=pd.DataFrame(allnormal,columns=column_names)
df_abnormal=pd.DataFrame(allabnormal, columns=column_names)
# print(df_normal.shape)
# print(df_abnormal.shape)
#df_combo=pd.concat([df_normal, df_abnormal], ignore_index=True)

combo=np.vstack((allnormal,allabnormal)) # with numpy ..no dataframe used
#print(combo)


#%%

## check for min and max distribution lengths in expected and observed coverage

m=[len(e) for e in df_combo["expected_coverage"]]
print(max(m))
s=[len(e) for e in df_abnormal["expected_coverage"]]
print(min(s))
# n=[e for e in df_abnormal["transcript_names"]]
# print(len(n))
l=[len(e) for e in df_normal["observed_coverage"]]
print(min(l))

#exit()


# plot graph for transcript filtering for maxlen=5000
import matplotlib.pyplot as plt

plt.plot(m)
plt.xlabel("Total Transcripts")
plt.ylabel(" Coverage Length ")
plt.title("Transcripts vs. Coverage Lengths ")
# draw vertical line from (70,100) to (70, 250)
plt.plot([0, 150240], [5000, 5000], 'k-', lw=2)
plt.show()




#%%
#filter transcripts 
combo_new=[]
maxlen=5000
for i in range(combo.shape[0]):
    if combo[i,1].shape[0] <= maxlen:
        combo_new.append([combo[i,1],combo[i,2],combo[i,3]])


combo_new=np.array(combo_new)
#print(combo_new.shape)



# pad sequences to maxlen 
from keras.preprocessing.sequence import pad_sequences

#e_o= pad_sequences(combo[:, 1:3], padding='post', dtype='float32')
e= pad_sequences(combo_new[:, 0], padding='post',maxlen=maxlen,dtype='float32')
o= pad_sequences(combo_new[:, 1], padding='post',maxlen=maxlen,dtype='float32')

# print(e.shape)
# print(o.shape)
# print(e)
# print(o)
X_combo=np.stack((e,o),axis=-1) # out of which test/train/val will be created
print(X_combo)
print(X_combo.shape)



#%%

# change object type with label encoder 
y_combo=combo_new[:,-1]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_combo_labelcode=le.fit_transform(y_combo)
# print(y_combo_labelcode.dtype)
# print(y_combo_labelcode)




#%%
# split dataset into test and train
from sklearn.model_selection import train_test_split

X_combo_train, X_combo_test, y_combo_train,y_combo_test=train_test_split(X_combo,y_combo_labelcode, train_size=0.9, random_state=42)



#%%

# KERAS METHOD
import numpy
import tensorflow as tf

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional

# fix random seed for reproducibility
numpy.random.seed(42)
tf.random.set_seed(42)


model = Sequential()
model.add(Bidirectional(LSTM(100), input_shape=(X_combo[0].shape)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%%
# further split train with validation
history=model.fit(X_combo_train, y_combo_train, epochs=30, batch_size=20, verbose=1,validation_split=0.2)


#model.save("/home/priyamvada/new/savemodel/model2final1")


#%%

#print(history.history.keys())

#%%
import matplotlib.pyplot as plt
#validation set :It's purpose is to track progress through validation loss and accuracy.
# plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%%
# model evaluated on test set with accuracy output 
test_results = model.evaluate(X_combo_test, y_combo_test, verbose=False)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')







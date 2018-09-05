import pandas as pd
import numpy as np

path_train = 'data/train/train.csv'

data = pd.read_csv(path_train, low_memory=False)

train_data = data.values[:, 0:-2]
train_label = data.values[:, -2].tolist()
user_id = data.values[:, -1].tolist()

Y_label = train_label
label = []
for i in Y_label:
    if i not in label :
        label.append(i)
Y = []
for j in Y_label:
    for k in range(len(label)):
        if j == label[k]:
            Y.append(k+1)

print(label)
print(Y_label[0:10])
print(Y[0:10])


import numpy as np
a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])
print(np.argmax(a, axis=1))
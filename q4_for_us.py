import pandas as pd

train_set = pd.read_csv('train.csv', sep=',', header=None)
r_c = 0
w_c = 0
print(train_set)
train_set = train_set.to_numpy()
for line in train_set:
    print(line[0])
    if line[0] == 'M':
        r_c += 1
    else:
        w_c += 1

pres = r_c/(r_c+8*w_c)
print(pres)

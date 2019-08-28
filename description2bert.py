import pandas as pd
from bert_juman import BertWithJumanModel
import numpy as np
import pandas as pd

bert = BertWithJumanModel("../Japanese_L-12_H-768_A-12_E-30_BPE")


df = pd.read_csv('./rurubu.csv',delimiter=',')
a1 = []

data = np.empty((1,768))
temp = np.array([])
i = 0
for index,item in  df.iterrows():
    value = item['event_name'] + ' ' + item['description']
    print( value )
    temp = bert.get_sentence_embedding(value)
    if i == 0:
        data = temp.reshape(1,768)
        i+=1 
        continue
    else :
        temp = temp.reshape(1,768)
        data = np.append(data, temp,axis=0)
        i+=1

    print('test' + str(i))

np.savetxt(fname="out_description.tsv", X = data ,delimiter =',')
np.save('out_description',data)
#data.to_csv('output.tsv', sep='\t', index=False)

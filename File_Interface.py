__author__ = 'multiangle'

import csv,pickle

def read_csv(path):         #读入原始csv文件，不做任何变动
    file=open(path,'r')
    reader=csv.reader(file)
    data=[row for row in reader]
    return data
def load_pickle(path):          #读入pickle文件，不做任何变动
    file=open(path,'rb')
    data=pickle.load(file)
    file.close()
    return data
def save_pickle(data,path):
    file=open(path,'wb')
    pickle.dump(data,file)
    file.close()
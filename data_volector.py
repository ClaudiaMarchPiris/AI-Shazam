import numpy as np
import random
import pickle

def pikeling_samples(size,array,amount):
     sequens = 'data files/sequens'
     infile = open(sequens,'rb')
     integer = pickle.load(infile)
     infile.close()
     for i in reange(amount):
         pikeling_place = 'data files/data number'+str(i+integer)
         outfile = open(pikeling_place,'wb')
         pickle.dump(cut_inteval_random(size,array),outfile)
         outfile.close()
     sequensout = open(sequens,'wb')
     pickle.dump(integer+size,sequensout)
     sequensout.close()
def cut_inteval_random(interval,array):
    return array(-interval:random.randint(interval,array.leangt-1)

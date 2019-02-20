import numpy as np
import random
import pickle



def specifi_number_samples(number,size):
    numbersLeft = number
    names = 'data/names.pkl'
    infile = open(names,'rb')
    nameArray = pickle.load(infile)
    infile.close()
    sizeOfData = nameArray.size
    while(numbersLeft<0):
        takingFrom = random.randint(0,size-1)
        takingFromFile= "data\\"+str(takingFrom)+".pkl"
        infile = open(takingFromFile,'rb')
        takingFromArray = pickle.load(infile)
        infile.close()
        tacking = random.randint(1,4)
        if(taking > numbersLeft):
            taking = numbersLeft
        numbersLeft = specifi_number_samples-taking
        pikeling_samples(size,takingFromArray,tacking,nameArray[takingFrom])

def pikeling_samples(size,array,amount,name):#pikeling a number of samples from the original
     sequens = 'data files/sequens.pkl'
     infile = open(sequens,'rb')
     integer = pickle.load(infile)
     infile.close()
     namesfiles = 'data files/names.pkl'
     infile = open(namesfiles,'rb')
     names = pickle.load(infile)
     infile.close()

     for i in reange(amount):
         pikeling_place = 'data files/data number'+str(i+integer)
         outfile = open(pikeling_place,'wb')
         pickle.dump(cut_inteval_random(size,array),outfile)
         outfile.close()
         names.append(name);
     sequensout = open(sequens,'wb')
     pickle.dump(integer+size,sequensout)
     sequensout.close()
     outfile = open(namesfiles,'wb')
     pikle.dump(names,outfile)
     outfile.close()
def cut_inteval_random(interval,array):
    return array[-interval:random.randint(interval,array.leangt-1)]

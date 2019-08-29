#Importing the packages
from random import seed
from random import randrange
from csv import reader
from math import sqrt

#Load a csv file
def load_csv(filename):
    dataset=list()
    with open(filename, 'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Convert string column to float
def str_column_to_float(dataset,column):
    for row in dataset:
        row[column]=float(row[column].strip())
def str_column_to_int(dataset,column):
    class_values=[row[column] for row in dataset]
    unique=set(class_values)
    lookup=dict()
    for i,value in enumerate(unique):
        lookup[value]=i
    for row in dataset:
        row[column]=lookup[row[column]]
    return lookup

#Split a dataset into k folds
def cross_validation_split(dataset,n_folds):
    dataset_split=list()
    dataset_copy=list(dataset)
    fold_size=int(len(dataset_copy))
    for i in range(n_folds):
        fold=list()
        while len(fold) < fold_size:
            index=randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

#Calculate the accuracy percentage
def accuracy_metric(actual,predicted):
    correct=0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            coorect+=1
    return coorect/float(len(actual)) * 100.0

#Evaluate and algorithm using a cross validation split
def evaluate_algorithm(dataset,algorithm,n_folds,*args):
    folds=cross_validation_split(dataset,n_folds)
    scores=list()
    for fold in folds:
        train_set=list(folds)
        train_set.remove(fold)
        train_set=list()
        for row in fold:
            row_copy=list(row)
            test_set.append(row_copy)
            row_copy[-1]=none
        predicted=algorithm(train_set,test_set,*args)
        actual=[row[-1] for row in fold]
        accuracy=accuracy_metric(actual,predicted)
        scores.append(accuracy)
    return scores


#Split a dataset based on an attribute and an attribute value
def test_split(index,value,dataset):
    left,right=list(),list()
    for row in dataset:
        if row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
    return left,right



#Calculate the Gini index for a split dataset
def gini_index(groups,classes):
    n_instances=float(sum[len(group) for group in groups])
    gini=0.0
    for 






















            

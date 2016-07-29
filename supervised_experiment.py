from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import metrics
import pickle
import numpy as np
import random
import time

min_terms_preprocess = 10000
dataset = 'all_data_' + str(min_terms_preprocess)+ '.pickle'
all_data = pickle.load(open(dataset,"r"))
file= open('supervised_experiment'+ str(min_terms_preprocess)+ '.txt','w')
print "Random Forest is loading..."
random.shuffle(all_data)
n_samples = len(all_data)
X = []
y = []
for value,label in all_data:
	X.append(value)
	y.append(label)
X = np.array(X)
y = np.array(y)
kf = KFold(n_samples, n_folds=10)
k = 0
start_time = time.time()

for train, test in kf:
    rfc = RandomForestClassifier(n_estimators=100)	
    rfc.fit(X[train], y[train])
    predicted = rfc.predict(X[train])
    if (kf == 10):
	start_time = time.time() # doesn't work because scoping
   # file.write(str("Classification report for classifier %s \n%s\n" % (rfc, metrics.classification_report(y[test], predicted))))
    file.write(str(rfc.score(X[test],y[test])))
    file.write(str("\n"))
    k+=1

file.write(str("Time it took to execute: " + str(time.time() - start_time) + " seconds."))      
print "Random Forest Output presented in the .txt file"    
print "Loading next classifier..."


# Load the preprocessed data
all_data = pickle.load(open(dataset,"r"))
print 'The Support Vector Machine is training ...' + str(min_terms_preprocess)
# initialize varables
n_samples = len(all_data)
X = []
y = []
# Partition the preprocessed data into labels and values(samples)
for value,label in all_data:
	X.append(value)
	y.append(label)
# convert lists into arrays 
X = np.array(X)
y = np.array(y)
# K-Fold cross validation 
kf = KFold(n_samples, n_folds=10, shuffle=True)
k = 0
#Build the classifier
classifier = svm.SVC(gamma=0.001).fit(X, y)
start_time = time.time()
#For each Kfold attempt to predict new values based on previous training
for train, test in kf:
    predicted = classifier.predict(X[test])
    if (kf == 10):
	start_time = time.time()
    file.write(str("Classification report for classifier %s for fold %d:\n%s\n" % (classifier, k, metrics.classification_report(y[test], predicted))))
    k+=1
file.write(str("Time it took to execute: " + str(time.time() - start_time) + " seconds."))   
print "Output displayed in txt file"
file.close 

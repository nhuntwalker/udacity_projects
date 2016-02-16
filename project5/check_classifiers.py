#check the data
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as sql
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans, MeanShift
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
    BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split


with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

x_train = data["x_train"]
x_test = data["x_test"]
y_train = data["y_train"]
y_test = data["y_train"]

def plot_precision_recall_param(precision, recall, param, param_name):
    """
    This function takes arrays of precision and recall scores, as well as
    the array of parameter choices, then simply plots score vs parameter value.
    The max of each score will be pointed out by a downward-pointing triangle.

    This is specifically for algorithm tuning, to visualize the landscape of
    tuning results. It does NOT output the optimal parameter value. It just
    shows what was obtained.
    """
    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(111)
    ax.plot(param, precision, color="b", label="precision")
    ax.scatter(param[precision.argmax()], precision[precision.argmax()] + 0.01, 
               marker="v", c="r")
    ax.text(0.02, 0.9, 
            "Precision Max: %.3f at %s: %g" % (precision.max(), param_name, 
                                               param[precision.argmax()]),
            horizontalalignment="left", verticalalignment="top", 
            transform=ax.transAxes)
    
    ax.plot(param, recall, color="g", label="recall")
    ax.scatter(param[recall.argmax()], recall[recall.argmax()] + 0.01, 
               marker="v", c="k")
    ax.text(0.02, 0.8, 
            "Recall Max: %.3f at %s: %g" % (recall.max(), param_name, 
                                            param[recall.argmax()]),
            horizontalalignment="left", verticalalignment="top", 
            transform=ax.transAxes)
    
    ax.set_xlabel(param_name)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.minorticks_on()
    ax.legend()
    plt.show()

## Tuning K-Means
iter_array = np.arange(1, 300)
prec = np.zeros(len(iter_array))
rec = np.zeros(len(iter_array))

for ii in range(len(iter_array)):
    clf = KMeans(n_clusters=2, max_iter = iter_array[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred)
    rec[ii] = recall_score(y_test, pred)

plot_precision_recall_param(prec, rec, iter_array, "max_iter")


init_array = np.arange(1, 100)
prec = np.zeros(len(init_array))
rec = np.zeros(len(init_array))

for ii in range(len(init_array)):
    clf = KMeans(n_clusters=2, n_init = init_array[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred)
    rec[ii] = recall_score(y_test, pred)

plot_precision_recall_param(prec, rec, init_array, "init")


tol_array = np.logspace(-8, -2, 100)
prec = np.zeros(len(tol_array))
rec = np.zeros(len(tol_array))

for ii in range(len(tol_array)):
    clf = KMeans(tol = tol_array[ii], n_clusters=2, n_init=3)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred)
    rec[ii] = recall_score(y_test, pred)

plot_precision_recall_param(prec, rec, tol_array, "tol")


rand = np.arange(1, 100)
prec = np.zeros(len(rand))
rec = np.zeros(len(rand))

for ii in range(len(rand)):
    clf = KMeans(random_state=rand[ii], n_clusters=2, n_init=3, tol=1E-8)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred)
    rec[ii] = recall_score(y_test, pred)

plot_precision_recall_param(prec, rec, rand, "random_state")

## Tuning DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="gini", random_state=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = DecisionTreeClassifier(criterion="entropy", random_state=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = DecisionTreeClassifier(splitter="random", random_state=5, 
                             criterion="entropy")
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


feats = np.arange(1, len(features_list))
prec = np.zeros(len(feats))
rec = np.zeros(len(feats))

for ii in range(len(feats)):
    clf = DecisionTreeClassifier(splitter="random", max_features=feats[ii], 
                                 random_state=5, criterion="entropy")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, feats, "max_features")


depths = np.arange(1, 100)
prec = np.zeros(len(depths))
rec = np.zeros(len(depths))

for ii in range(len(depths)):
    clf = DecisionTreeClassifier(splitter="random", max_features=3, 
                                 max_depth=depths[ii], random_state=5,
                                criterion="entropy")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, depths, "max_depths")


min_samples = np.arange(2, 100)
prec = np.zeros(len(min_samples))
rec = np.zeros(len(min_samples))

for ii in range(len(min_samples)):
    clf = DecisionTreeClassifier(splitter="random", max_features=3, 
                                 max_depth=10, random_state=5, 
                                 criterion="entropy",
                                 min_samples_split=min_samples[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, min_samples, "min_samples_split")


min_samples = np.arange(1, 100)
prec = np.zeros(len(min_samples))
rec = np.zeros(len(min_samples))

for ii in range(len(min_samples)):
    clf = DecisionTreeClassifier(splitter="random", max_features=3, 
                                 max_depth=10, random_state=5, 
                                 criterion="entropy",
                                 min_samples_leaf=min_samples[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, min_samples, "min_samples_leaf")


rand = np.arange(1, 100)
prec = np.zeros(len(rand))
rec = np.zeros(len(rand))

for ii in range(len(rand)):
    clf = DecisionTreeClassifier(splitter="random", max_features=3, 
                                 max_depth=10, random_state=rand[ii], 
                                 criterion="entropy")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, rand, "random_state")

## Tuning LinearSVC
C_vals = np.logspace(-2, 2, 100)
prec = np.zeros(len(C_vals))
rec = np.zeros(len(C_vals))

for ii in range(len(C_vals)):
    clf = LinearSVC(C = C_vals[ii], random_state=5)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, C_vals, "C")


clf = LinearSVC(C=20, loss="hinge", random_state=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = LinearSVC(C=20, loss="squared_hinge", random_state=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = LinearSVC(C=20, random_state=5, dual=True)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = LinearSVC(C=20, random_state=5, dual=False)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = LinearSVC(C=20, random_state=5, fit_intercept=False)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")

clf = LinearSVC(C=20, random_state=5, fit_intercept=True)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


tol_array = np.logspace(-10, 0, 1000)
prec = np.zeros(len(tol_array))
rec = np.zeros(len(tol_array))

for ii in range(len(tol_array)):
    clf = LinearSVC(C=20, tol= tol_array[ii], random_state=5,
                   fit_intercept=False)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, tol_array, "tol")


## Tuning AdaBoostClassifier
n_estimators = np.arange(1, 200)
prec = np.zeros(len(n_estimators))
rec = np.zeros(len(n_estimators))

for ii in range(len(n_estimators)):
    clf = AdaBoostClassifier(n_estimators=n_estimators[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, n_estimators, "n_estimators")


learning_rate = np.linspace(0.01, 3, 100)
prec = np.zeros(len(learning_rate))
rec = np.zeros(len(learning_rate))

for ii in range(len(learning_rate)):
    clf = AdaBoostClassifier(learning_rate=learning_rate[ii], 
                             n_estimators=50)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, learning_rate, "learning_rate")


## Tuning BaggingClassifier
n_estimators = np.arange(1, 100)
prec = np.zeros(len(n_estimators))
rec = np.zeros(len(n_estimators))

for ii in range(len(n_estimators)):
    clf = BaggingClassifier(n_estimators=n_estimators[ii], 
                            warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, n_estimators, "n_estimators")


samples = np.arange(1, 50)
prec = np.zeros(len(samples))
rec = np.zeros(len(samples))

for ii in range(len(samples)):
    clf = BaggingClassifier(n_estimators=4, max_samples=samples[ii],
                           warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, samples, "max_samples")


feats = np.arange(1, len(features_list))
prec = np.zeros(len(feats))
rec = np.zeros(len(feats))

for ii in range(len(feats)):
    clf = BaggingClassifier(n_estimators=4, max_samples=35,
                           max_features=feats[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, feats, "max_features")

# Tuning GMM
tol_array = np.logspace(-8, -2, 100)
prec = np.zeros(len(tol_array))
rec = np.zeros(len(tol_array))

for ii in range(len(tol_array)):
    clf = GMM(tol=tol_array[ii], n_components=2)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, tol_array, "tol")


iters = np.arange(1, 50)
prec = np.zeros(len(iters))
rec = np.zeros(len(iters))

for ii in range(len(iters)):
    clf = GMM(tol=5E-3, n_components=2, n_iter=iters[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, iters, "n_iter")


inits = np.arange(1, 50)
prec = np.zeros(len(inits))
rec = np.zeros(len(inits))

for ii in range(len(inits)):
    clf = GMM(tol=5E-3, n_components=2, n_init=inits[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, inits, "n_init")''







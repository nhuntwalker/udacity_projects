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

## Tuning ExtraTrees
est_array = np.arange(1, 100)
prec = np.zeros(len(est_array))
rec = np.zeros(len(est_array))

for ii in range(len(est_array)):
    clf = ExtraTreesClassifier(n_estimators=est_array[ii], random_state=5)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, est_array, "n_estimators")


clf = ExtraTreesClassifier(criterion="entropy", n_estimators=32, random_state=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


clf = ExtraTreesClassifier(criterion="gini", n_estimators=32, random_state=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


feats = np.arange(1, len(features_list))
prec = np.zeros(len(feats))
rec = np.zeros(len(feats))

for ii in range(len(feats)):
    clf = ExtraTreesClassifier(n_estimators=32, max_features=feats[ii],
                               random_state=5)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, feats, "max_features")


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


clf = DecisionTreeClassifier(splitter="best", random_state=5, 
                             criterion="entropy")
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
    clf = DecisionTreeClassifier(splitter="random", max_features=4, 
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
    clf = DecisionTreeClassifier(splitter="random", max_features=4, 
                                 max_depth=20, random_state=5, 
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
    clf = DecisionTreeClassifier(splitter="random", max_features=4, 
                                 max_depth=20, random_state=5, 
                                 criterion="entropy", min_samples_split=5,
                                 min_samples_leaf=min_samples[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")
 
plot_precision_recall_param(prec, rec, min_samples, "min_samples_leaf")


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


tol_array = np.logspace(-6, 0, 1000)
prec = np.zeros(len(tol_array))
rec = np.zeros(len(tol_array))

for ii in range(len(tol_array)):
    clf = LinearSVC(C=20, loss="squared_hinge", random_state=5, dual=True, 
                    fit_intercept=True, tol=tol_array[ii])
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
    clf = AdaBoostClassifier(n_estimators=n_estimators[ii], learning_rate=2.2)
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
                             n_estimators=7)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, learning_rate, "learning_rate")


## Tuning GradientBoostingClassifier
clf = GradientBoostingClassifier(loss="deviance", warm_start=True)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


clf = GradientBoostingClassifier(loss="exponential", warm_start=True)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


learn = np.linspace(1, 35, 500)
prec = np.zeros(len(learn))
rec = np.zeros(len(learn))

for ii in range(len(learn)):
    clf = GradientBoostingClassifier(learning_rate=learn[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, learn, "learning_rate")


estimators = np.arange(1, 50)
prec = np.zeros(len(estimators))
rec = np.zeros(len(estimators))

for ii in range(len(estimators)):
    clf = GradientBoostingClassifier(learning_rate=8., n_estimators=estimators[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, estimators, "n_estimators")


depths = np.arange(1, 50)
prec = np.zeros(len(depths))
rec = np.zeros(len(depths))

for ii in range(len(depths)):
    clf = GradientBoostingClassifier(learning_rate=8., n_estimators=5, 
                                     max_depth=depths[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, depths, "max_depth")


splits = np.arange(2, 50)
prec = np.zeros(len(splits))
rec = np.zeros(len(splits))

for ii in range(len(splits)):
    clf = GradientBoostingClassifier(learning_rate=8., n_estimators=5, max_depth=11,
                                     min_samples_split=splits[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, splits, "min_samples_split")


samples = np.arange(2, 50)
prec = np.zeros(len(samples))
rec = np.zeros(len(samples))

for ii in range(len(samples)):
    clf = GradientBoostingClassifier(learning_rate=8., n_estimators=5, max_depth=11,
                                     min_samples_split=14, warm_start=True, 
                                     min_samples_leaf=samples[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, samples, "min_samples_leaf")


samples = np.linspace(0.7, 1.0, 50)
prec = np.zeros(len(samples))
rec = np.zeros(len(samples))

for ii in range(len(samples)):
    clf = GradientBoostingClassifier(learning_rate=8., n_estimators=5, max_depth=11,
                                     min_samples_split=14, warm_start=True, 
                                     min_samples_leaf=9, subsample=samples[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, samples, "subsample")


feats = np.arange(1, len(features_list))
prec = np.zeros(len(feats))
rec = np.zeros(len(feats))

for ii in range(len(feats)):
    clf = GradientBoostingClassifier(learning_rate=8., n_estimators=5, max_depth=11,
                                     min_samples_split=14, warm_start=True, 
                                     min_samples_leaf=9, max_features=feats[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, feats, "max_features")


## Tuning RandomForest Classifier
clf = RandomForestClassifier(criterion="entropy", warm_start=True)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


clf = RandomForestClassifier(criterion="gini", warm_start=True)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print "Precision Score: %.3f" % precision_score(y_test, pred, average="binary")
print "Recall Score: %.3f" % recall_score(y_test, pred, average="binary")


estimators = np.arange(1, 50)
prec = np.zeros(len(estimators))
rec = np.zeros(len(estimators))

for ii in range(len(estimators)):
    clf = RandomForestClassifier(n_estimators=estimators[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, estimators, "n_estimators")


feats = np.arange(1, len(features_list))
prec = np.zeros(len(feats))
rec = np.zeros(len(feats))

for ii in range(len(feats)):
    clf = RandomForestClassifier(n_estimators=18, max_features=feats[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, feats, "max_features")


depths = np.arange(1, 50)
prec = np.zeros(len(depths))
rec = np.zeros(len(depths))

for ii in range(len(depths)):
    clf = RandomForestClassifier(n_estimators=18, max_features=2,
                                 max_depth=depths[ii], warm_start=True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, depths, "max_depth")


splits = np.arange(2, 50)
prec = np.zeros(len(splits))
rec = np.zeros(len(splits))

for ii in range(len(splits)):
    clf = RandomForestClassifier(n_estimators=18, max_features=2,
                                 max_depth=7, warm_start=True, min_samples_split=splits[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, splits, "min_samples_split")


samples = np.arange(1, 50)
prec = np.zeros(len(samples))
rec = np.zeros(len(samples))

for ii in range(len(samples)):
    clf = RandomForestClassifier(n_estimators=18, max_features=2,
                                 max_depth=7, warm_start=True, min_samples_leaf=samples[ii])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prec[ii] = precision_score(y_test, pred, average="binary")
    rec[ii] = recall_score(y_test, pred, average="binary")

plot_precision_recall_param(prec, rec, samples, "min_samples_leaf")

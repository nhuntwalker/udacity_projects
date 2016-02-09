#!/usr/bin/python

import sys
import pickle
sys.path.append("../../courses/machine-learning/ud120-projects/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import pandasql as sql
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', "total_earned_cash_value", 
                "exercised_stock_options"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Two oddities to be dealt with. Robert Belfer's data is incorrectly
# input, as is Sanjay Bhatnagar's.
data_dict["BELFER ROBERT"]["deferred_income"] = -102500
data_dict["BELFER ROBERT"]["deferral_payments"] = 0
data_dict["BELFER ROBERT"]["total_payments"] = 3285
data_dict["BELFER ROBERT"]["exercised_stock_options"] = 0
data_dict["BELFER ROBERT"]["restricted_stock"] = 44093
data_dict["BELFER ROBERT"]["restricted_stock_deferred"] = -44093
data_dict["BELFER ROBERT"]["total_stock_value"] = 0
data_dict["BELFER ROBERT"]["expenses"] = 3285
data_dict["BELFER ROBERT"]["director_fees"] = 102500

data_dict["BHATNAGAR SANJAY"]["expenses"] = 137864
data_dict["BHATNAGAR SANJAY"]["total_payments"] = 137864
data_dict["BHATNAGAR SANJAY"]["total_stock_value"] = 15456290
data_dict["BHATNAGAR SANJAY"]["exercised_stock_options"] = 15456290
data_dict["BHATNAGAR SANJAY"]["restricted_stock"] = 2604490
data_dict["BHATNAGAR SANJAY"]["restricted_stock_deferred"] = -2604490
data_dict["BHATNAGAR SANJAY"]["other"] = 0
data_dict["BHATNAGAR SANJAY"]["director_fees"] = 0

# I'm transforming the data into a pandas dataframe for ease of use for me
data_df = pd.DataFrame.from_dict(data_dict, orient="index")
data_df.replace("NaN", 0, inplace=True)
data_df.poi = data_df.poi.astype(bool)
data_df.email_address = data_df.email_address.astype("str")
data_df["employee"] = data_df.index

### Task 2: Remove outliers
# Outliers in this context would be (for my purposes) employees with no
# reported salary, no reported total payments, and no total stock value.
# I expect my POI classifier will rely on at least one of these (or one of
# the components that will contribute to one of the totals), so if all of
# these are null, then they're not useful
the_query = "SELECT *"
the_query += " FROM data_df"
the_query += " WHERE total_payments!=0 AND salary!=0 AND total_stock_value!=0"
no_null_df = sql.sqldf(the_query, locals())

# Remove some 3-sigma outliers. Salary is getting a 2-sigma cut because
# 3-sigma doesn't remove an outlier that skews my sample.
column_lims = {
    "bonus" : no_null_df.bonus.quantile(0.995),
    "salary" : no_null_df.salary.quantile(0.95),
    "person2poi" : no_null_df.from_this_person_to_poi.quantile(0.995),
    "longterm" : no_null_df.long_term_incentive.quantile(0.995),
    "allstock" : no_null_df.total_stock_value.quantile(0.995)
}

three_sig_cut = (no_null_df.bonus < column_lims["bonus"]) & \
    (no_null_df.salary < column_lims["salary"]) & \
    (no_null_df.from_this_person_to_poi < column_lims["person2poi"]) & \
    (no_null_df.long_term_incentive < column_lims["longterm"]) & \
    (no_null_df.total_stock_value < column_lims["allstock"])

no_outliers = no_null_df[three_sig_cut]

### Task 3: Create new feature(s)
# I'm rescaling a few properties as the point-separation appears more clearly
# that way.
no_outliers["log_salary"] = np.log10(no_outliers.salary + 1)
no_outliers["log_to_messages"] = np.log10(no_outliers.to_messages + 1)
no_outliers["log_total_payments"] = np.log10(no_outliers.total_payments + 1)
no_outliers["log_bonus"] = np.log10(no_outliers.bonus + 1)
no_outliers["log_restricted_stock"] = np.log10(no_outliers.restricted_stock + 1)
no_outliers["log_total_stock_value"] = \
    np.log10(no_outliers.total_stock_value + 1)

no_outliers["log_other"] = np.log10(no_outliers.other + 1)
no_outliers["log_long_term_incentive"] = \
    np.log10(no_outliers.long_term_incentive + 1)

no_outliers["log_expenses"] = np.log10(no_outliers.expenses + 1)

# stocks_cash_ratio
col = "stocks_cash_ratio"
no_outliers[col] = no_outliers.total_stock_value / no_outliers.total_payments

# total_emails, fractions_sent, fractions_received
no_outliers["total_emails"] = no_outliers[["from_messages", 
                                           "to_messages"]].sum(axis=1)
col = "fractions_sent"
no_outliers[col] = no_outliers.from_messages / \
    no_outliers.total_emails.astype(float)

### Store to my_data for easy export below.
my_data = no_outliers.fillna(0).to_dict(orient="index")

for key in my_data:
    the_name = my_data[key]["employee"]
    my_data[the_name] = my_data.pop(key)

### Extract features and labels from dataset for local testing
# Rewriting this so that it works with my dataframe
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)
features_list = ["poi", "log_other", "stocks_cash_ratio", "fractions_sent", 
                 "log_expenses"]
data = featureFormat(my_data, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def evaluate_classifiers(classifier_dict, features, labels, iters=50, 
                         prec_avg=None, plot=False):
    """
    Pass a dict of classifiers and the data that you desire to be classified.
    This function will test those classifiers against that data, each time
    testing against a train_test_split with a 60-40 ratio. 

    If 'plot' is set to False, then this function will print the mean, min, 
    median, and max for the accuracy score, precision score, and recall score 
    for each classifier. 

    If 'plot' is set to True, then will produce a 3-panel plot for the 
    accuracy, precision, and recall scores for each run. I don't recommend
    doing this unless the number of classifiers in classifier_dict is less <= 5.
    """
    np.random.seed(42) #life, the universe, and everything
    
    if plot:
        fig = plt.figure(figsize=(10,10))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        
    for classifier in classifier_dict.keys():
        clf = classifier_dict[classifier]
        acc_scores = np.zeros(iters)
        prec_scores = np.zeros(iters)
        recall_scores = np.zeros(iters)

        for ii in range(iters):
            X_train, X_test, y_train, y_test = train_test_split(features, 
                                                                labels, 
                                                                test_size=0.4)

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc_scores[ii] = accuracy_score(y_test, pred)

            if prec_avg:
                prec_scores[ii] = precision_score(y_test, pred, 
                                                  average=prec_avg)
                recall_scores[ii] = recall_score(y_test, pred, 
                                                 average=prec_avg)

            else:
                prec_scores[ii] = precision_score(y_test, pred)
                recall_scores[ii] = recall_score(y_test, pred)
              

        if plot:
            quant_str = ": %.3f $\pm$ %.3f"
            ax1.plot(acc_scores,
                     label=classifier + quant_str % (acc_scores.mean(),
                                                     acc_scores.std()))
            ax2.plot(prec_scores,
                     label=classifier + quant_str % (prec_scores.mean(),
                                                     prec_scores.std()))
            ax3.plot(recall_scores,
                     label=classifier + quant_str % (recall_scores.mean(),
                                                     recall_scores.std()))
            
        else:
            acc_str = "\tAccuracy - %.6f (%.6f, %.6f, %.6f)"
            prec_str = "\tPrecision - %.6f (%.6f, %.6f, %.6f)"
            rec_str = "\tRecall - %.6f (%.6f, %.6f, %.6f)\n"

            print "%s: " % classifier
            print acc_str % (acc_scores.mean(), acc_scores.min(),
                             np.median(acc_scores), acc_scores.max())

            print prec_str % (prec_scores.mean(), prec_scores.min(),
                              np.median(prec_scores), prec_scores.max())

            print rec_str % (recall_scores.mean(), recall_scores.min(),
                             np.median(recall_scores), recall_scores.max())
    
    if plot:
        ax1.set_ylabel("Accuracy Score")
        ax2.set_ylabel("Precision Score")
        ax3.set_ylabel("Recall Score")
        ax1.set_ylim(0, 1.5)
        ax2.set_ylim(0, 1.5)
        ax3.set_ylim(0, 1.5)
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        ax3.legend(fontsize=10)
        plt.show()

# Provided to give you a starting point. Try a variety of classifiers.
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

classifiers = {"GaussianNB" : GaussianNB(), 
               "Decision Tree" : DecisionTreeClassifier(random_state=5),
               "KNearest" : KNeighborsClassifier(),
               "SVC - rbf" : SVC(kernel="rbf"),
               "LinearSVC" : LinearSVC(),
               "KMeans" : KMeans(random_state=5, n_clusters=2),
               "RandomForestClassifier" : RandomForestClassifier(),
               "RandomForestClassifier - entropy" : \
                    RandomForestClassifier(criterion='entropy'),
               "AdaBoostClassifier" : AdaBoostClassifier(),
               "BaggingClassifier" : BaggingClassifier(),
               "ExtraTreesClassifier" : ExtraTreesClassifier(),
               "GradientBoostingClassifier" : GradientBoostingClassifier(),
               "RidgeClassifier" : RidgeClassifier()}

evaluate_classifiers(classifiers, features, labels)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
x_train, x_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size = 0.4, 
                                                    random_state = 42)
data = {"x_train" : x_train, "y_train" : y_train, "x_test" : x_test, 
        "y_test" : y_test}

with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

## export to check_classifiers.py to test different classifiers and optimize

classifiers = {"KMeans" : KMeans(n_clusters=2, n_init=50, tol=1E-8),
               "LinearSVC" : LinearSVC(C=20.0, random_state=5, 
                                    fit_intercept=False),
               "GaussianNB" : GaussianNB(),
               "BaggingClassifier" : BaggingClassifier(max_samples=35, 
                                                       max_features=3, 
                                                       warm_start=True),
               "AdaBoostClassifier" : AdaBoostClassifier(learning_rate=0.85,
                                                         n_estimators=50),
               "DecisionTreeClassifier" : \
                DecisionTreeClassifier(splitter="random", 
                                     max_features=3, 
                                     max_depth=10, 
                                     random_state=40, 
                                     criterion="entropy")}

evaluate_classifiers(classifiers, features, labels, prec_avg="weighted", 
                     plot=True, iters=100)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = BaggingClassifier(max_samples=35, max_features=3, warm_start=True)
dump_classifier_and_data(clf, my_data, features_list)

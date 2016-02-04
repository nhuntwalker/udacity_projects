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
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['salary', "total_earned_cash_value", 
                "exercised_stock_options"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Two oddities to be dealt with. Robert Belfer's data is incorrectly
### input, as is Sanjay Bhatnagar's.
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

### I'm transforming the data into a pandas dataframe for ease of use for me
data_df = {
    "employee":[], "poi":[], "salary":[], "to_messages":[], 
    "deferral_payments":[], "total_payments":[], "exercised_stock_options":[],
    "bonus":[], "restricted_stock":[], "shared_receipt_with_poi":[],
    "restricted_stock_deferred":[], "total_stock_value":[], "expenses":[],
    "loan_advances":[], "from_messages":[], "other":[], 
    "from_this_person_to_poi":[], "director_fees":[], "deferred_income":[],
    "long_term_incentive":[], "email_address":[], "from_poi_to_this_person":[]
}

for employee_name in data_dict.keys():
    if employee_name != "TOTAL":
        data_df["employee"].append(employee_name)
        for key in data_df.keys():
            if key != "employee":
                if key in data_dict[employee_name].keys():
                    if data_dict[employee_name][key] == "NaN":
                        data_df[key].append(0)

                    else:
                        data_df[key].append(data_dict[employee_name][key])

                else:
                    data_df[key].append(0)


data_df = pd.DataFrame(data_df)
data_df.poi = data_df.poi.astype(bool)

### Task 2: Remove outliers
### Outliers in this context would be (for my purposes) employees with no
### reported salary, no reported total payments, and no total stock value.
### I expect my POI classifier will rely on at least one of these (or one of
### the components that will contribute to one of the totals), so if all of
### these are null, then they're not useful
the_query = "SELECT *"
the_query += " FROM data_df"
the_query += " WHERE total_payments!=0 AND salary!=0 AND total_stock_value!=0"
not_null_employees = sql.sqldf(the_query, locals())
data_df = not_null_employees

### For actual outliers in the data, I want to remove employees with
### bonus > 400,000, exercised_stock_options > 10 million, and salary > 500,000
data_df = data_df[(data_df.bonus < 4E6) & (data_df.salary < 500000) &
                  (data_df.exercised_stock_options < 1E7)]

# ### A different set of outliers would be those with no email addresses. These
# ### won't be useful to me if I intend to use email text to identifiy POIs
# the_query = "SELECT *"
# the_query += " FROM data_df"
# the_query += " WHERE email_address != 0"
# with_emails = sql.sqldf(the_query, locals())


### Task 3: Create new feature(s)
### The new feature to create will be "total_earned_cash_value", combining
### salary, bonus, director_fees, expenses, loan_advances, long_term_incentive
### deferral_payments / 0.9, and -(deferred_income)
data_df["total_earned_cash_value"] = data_df[["salary", "bonus", "director_fees", "expenses", "loan_advances", "long_term_incentive"]].sum(axis=1) + data_df.deferral_payments/0.9 - data_df.deferred_income
## stocks_cash_ratio
data_df["stocks_cash_ratio"] = data_df.total_stock_value / data_df.total_payments

#### Moving forward with data frames instead of dictionaries, so commenting these out
# back_to_dict = {}
# for name in data_df.employee:
#     back_to_dict[name] = {}
#     for col in data_df.columns:
#         if col != "employee":
#             if (col != "poi") & (col != "email_address"):
#                 back_to_dict[name][col] = float(data_df[data_df.employee == name][col])
#         
#             elif col == "email_address":
#                 back_to_dict[name][col] = str(data_df[data_df.employee == name][col]).split(" ")[4].split("\n")[0]
#
#             elif col == "poi":
#                 back_to_dict[name][col] = data_df[data_df.employee == name][col]
#
### Store to my_dataset for easy export below.
# my_dataset = data_dict

### Extract features and labels from dataset for local testing
#### Rewriting this so that it works with my dataframe
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)
labels = data_df.poi
features = data_df[features_list]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#### Let's start by finding some principal components amongst *all* features
features_list = ["bonus", "deferral_payments", "deferred_income", 
"exercised_stock_options", "expenses", "from_messages", "from_poi_to_this_person",
"from_this_person_to_poi", "long_term_incentive",
 "salary", "shared_receipt_with_poi",
"to_messages", "total_payments", "total_stock_value", "total_earned_cash_value",
"stocks_cash_ratio"]

indx = 1
fig = plt.figure(figsize=(12,8))
for feature in features_list:
    ax = plt.subplot(4, 4, indx)
    ax.hist(data_df[feature], bins=20)
    ax.text(0.9, 0.9, feature, transform=ax.transAxes,
        horizontalalignment="right", verticalalignment="top")
    indx += 1

plt.show()

from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=5, whiten=False).fit(features)
features_pca = pca.transform(features)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
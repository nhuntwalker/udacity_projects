#Enron Free-Response Questions

## Question 1

**Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?**

A: The goal of this project is to produce a classifier that can accurately classify actual persons of interest from the Enron data set, while minimizing the number of false negatives. There's also a secondary goal of producing potential persons of interest amongst the false positives, who have financial and communication characteristics similar to known persons of interest. For this, Machine Learning is a wonderful tool. By eye we can at best classify by pairs of characteristics at a time, and even then we have trouble combining the results of multiple pairs. With Machine Learning, we can assess all of the data from financial and communication realms together, finding non-obvious relationships that can help with our classification. We can then easily extend those relationships to the full employee list and to a fairly high degree accumulate a list of real and potential persons of interest.

The outliers that I'd found in the data set came in three varieties:

1. Incorrect inputs
2. Employees missing salary, stock, or total payments information
3. Employees above the 95th percentile in salary, and above the 99.5 percentile in bonuses, total stock value, long term incentives, and email counts from the employee to persons of interest

**The first set contained two outliers.** I encountered them while doing a simple univariate exploration of the data early on. They were simple but tedious to deal with. They required that I simply look directly at the financial data from the accompanying  `enron61702insiderpay.pdf` file, and rewrite the information in the raw data dictionary. The tedious part was making sure that there were no other employees with incorrectly-input data. These outliers were not removed from the data set, just reincorporated with the appropriate information. **The second set contained 55 outliers.** This is a fairly large number, representing about 37.7% of the initial data set, and they were omitted from the final analysis. However, it was necessary considering that one of the features that I considered to be key, feature that I created, involves all three of those figures. On a more human note, persons of interest in the case of Enron are money-motivated individuals. The likelihood of an employee being a person of interest while missing any of those three (really two, since salary is incorporated into total payments) characteristics is fairly low. That being said, the restriction on salary does omit one person of interest that made out like a bandit with stocks (Joseph Hirko; $30.7 million in stocks) because he had no salary information, along with 50 other employees not identified as persons of interest. **The third set contained 8 outliers** after the second set was removed. These were removed as they would drastically skew any classifier using these characteristics and searching for class-based means in the data. Out of these 8, three were persons of interest (David Delainey, Ken Lay, and Jeff Skilling). It makes sense to remove these from the data as they were at the very top of the company (and numerically above the 99.5 percentile), so other employees are not likely to have financial or communication information that's like theirs.


## Question 2

**What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.)**

A: The features that I ended up using were **log(other payments), the ratio of stocks to cash, the fraction of emails sent vs all emails for this person,** and **log(expenses)**. I first plotted histograms of potential characteristics of interest from the initial set to check what might need to be rescaled. Properties that spanned multiple orders of magnitude (i.e. most of the financial characteristics) were rescaled logarithmically. Then I plotted all the characteristics against each other to look for any natural separations between POI and non-POI. I then considered what relationships between characteristics might provide useful separations between employees based on what POIs might have experienced. This is when I created the stocks-to-cash ratio (total stock value divided by total payments) and then fractions of sent and received emails (sent/received emails divided by sent + received emails). After this point it became an iterative process of testing out classifiers with their default parameters on a set of potentially-useful characteristics, and then adding or removing characteristics as classifier performance changed.

## Question 3

**What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?**

A: I ended up using the BaggingClassifier, with `max_samples=35`, `max_features=3`, and `warm_start=True`, though I had come close to using the AdaBoostClassifier with `learning_rate=0.85` and `n_estimators=50` as its performance was pretty good on the set that I had, but when tested with `tester.py` it didn't perform as well.  I had tried a whole host of classifiers to start with using default parameters. I tested each classifier 50 times on pseudo-random train_test_splits of the data (test_size=0.4). I say pseudo-random because each set of 50 tests starts with a random seed of 42. I recorded the the min, median, max, and average performance of each classifier after the round of tests, looking for classifiers that were near or above the threshold of 0.3 in precision and recall scores. The table below shows my testing set, with the asterisked items as algorithms that I went on to tune:

| Classifier | Accuracy | Precision | Recall |
|------------|----------|-----------|--------|
|  | Avg (Min, Median, Max) | Avg (Min, Median, Max) | Avg (Min, Median, Max) |
| SVC | **0.841** (0.735, 0.852, 0.941) | **0.000** (0.000, 0.000, 0.000) | **0.000** (0.000, 0.000, 0.000) |
| RandomForest(entropy) | **0.832** (0.706, 0.824, 0.941) | **0.336** (0.000, 0.333, 1.000) | **0.139** (0.000, 0.155, 0.600) |
| Ridge | **0.832** (0.676, 0.824, 0.941) | **0.045** (0.000, 0.000, 1.000) | **0.016** (0.000, 0.000, 0.250) |
| LinearSVC* | **0.824** (0.706, 0.824, 0.912) | **0.388** (0.000, 0.367, 1.000) | **0.134** (0.000, 0.143, 0.400) |
| Bagging* | **0.812** (0.647, 0.824, 0.971) | **0.318** (0.000, 0.268, 1.000) | **0.211** (0.000, 0.200, 1.000) |
| RandomForest(gini) | **0.812** (0.676, 0.824, 0.941) | **0.220** (0.000, 0.000, 1.000) | **0.134** (0.000, 0.000, 1.000) |
| ExtraTrees | **0.803** (0.676, 0.794, 0.941) | **0.267** (0.000, 0.225, 1.000) | **0.148** (0.000, 0.167, 0.500) |
| KNearest | **0.802** (0.588, 0.794, 0.941) | **0.100** (0.000, 0.000, 1.000) | **0.023** (0.000, 0.000, 0.250) |
| AdaBoost* | **0.797** (0.647, 0.794, 0.882) | **0.400** (0.000, 0.268, 1.000) | **0.324** (0.000, 0.155, 1.000) |
| GradientBoosting | **0.784** (0.588, 0.794, 0.941) | **0.351** (0.000, 0.310, 1.000) |  **0.258** (0.000, 0.286, 0.600) |
| KMeans* | **0.771** (0.088, 0.824, 0.941) | **0.317** (0.000, 0.268, 1.000) | **0.210** (0.000, 0.155, 1.000) |
| DecisionTree* | **0.752** (0.529, 0.765, 0.882) | **0.279** (0.000, 0.250, 1.000) | **0.293** (0.000, 0.286, 0.750) |
| GaussianNB* | **0.642** (0.471, 0.662, 0.794) | **0.245** (0.000, 0.231, 0.750) | **0.530** (0.000, 0.536, 1.000) |
| GMM* | **0.689** (0.206, 0.691, 0.853) | **0.126** (0.000, 0.143, 0.375) | **0.190** (0.000, 0.167, 0.857) |

## Question 4

**What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?**

Tuning parameters refers to changing the values of the classifier's inputs (e.g. n_clusters in KMeans, C in SVC, or min_samples_leaf in DecisionTreeClassifier) to either make the classifier appropriate for the problem at hand, or to increase the performance of the classifier turn to the needs and nuances of the data. If you don't do this well, you can turn what would otherwise be a perfectly appropriate classifier into a completely useless machine that overbiases the model or overfits the data. 

I didn't tune just one algorithm. I tuned the KMeans, DecisionTree, LinearSVC, AdaBoost, Bagging, and GMM, as I wasn't convinced that that initial performance was the best that each of these classifiers could do. The Gaussian Naïve Bayes classifier was somewhat promising to start with but is entirely untuneable. For each of the classifiers I had tested, I began with a train-test split with test_size=0.4 and random_state=42. I then tested adjustments to parameters one at a time, calculating the precision and recall scores for each selection. I then visualized the results as line plots showing score vs parameter choice, and chose the parameter that produced either the best or near-best scores. I didn't prioritize either recall or precision, instead trying to choose options that minimized the loss of both as much as possible. I didn't always choose the best option, as oftentimes that would correspond to overfitting the data. Generally I tried to choose parameter values that produced peaks in precision/recall scores that didn't change sharply with small departures from that value.

## Question 5

**What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?**

Validation is a test of how an algorithm performs on different sub-selections of the data (that are hopefully in some way representative of the entire sample), aiming for somewhat consistent performance despite the differences in subsample. One classic mistake is to split data into training and testing sets *before* a min-max rescaling of the data or performing PCA, as those principal component fit is highly-dependent upon the input data. I validated my analysis by using `sklearn.cross_validation.train_test_split`, choosing my testing set as being 40% of my full data set. For each classifier that underwent tuning, I used train_test_split starting with a random seed of 42 over the same 100 "random" iterations and checked the performance of precision, accuracy, and recall.

## Question 6

**Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.**

I recorded the Accuracy, Precision, and Recall scores for each of my final classifiers, but only prioritized the latter two. I recorded these scores for 100 random train_test_splits of the data. For the Bagging Classifier that I eventually used, the average Accuracy | Precision | Recall scores were 0.896 ± 0.039 | 0.705 ± 0.213 | 0.596 ± 0.202. The rest of the algorithm performances are in the figure below.

The output statistics for my classifier mean the following: 

- **Accuracy:** on average, 89.6% (with little variance) of employees were classified matching their starting classifications
- **Precision:** on average, 70.5% of employees classified as persons of interest were actually known persons of interest
- **Recall:** 59.6% of actual persons of interest were classified as persons of interest.

All in all, even though not every person of interest fits the bill for this classifier, it does provide a healthy number of other employees that could be investigated without indicting the whole (or even the majority of the) company.
![this image](classifier_eval.png)
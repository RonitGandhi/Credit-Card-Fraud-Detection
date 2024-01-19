# Credit-Card-Fraud-Detection
Dataset Description
The dataset comprises of credit card transactions over a two-day period in September 2013,
with 492 fraud cases out of 284,807 transactions, resulting in a highly imbalanced dataset
(0.172% fraud instances). Features V1 to V28 are derived from PCA transformation, while
'Time' represents elapsed seconds since the first transaction, and 'Amount' denotes the
transaction amount. Due to confidentiality constraints, original features and additional
background information are undisclosed. The 'Class' feature, indicating fraud (1) or non-fraud
(0), serves as the response variable.

Data Preprocessing:
Synthetic Minority Over-sampling Technique, or SMOTE for short, is a preprocessing
technique used to address a class imbalance in a dataset.
In the real world, oftentimes we end up trying to train a model on a dataset with very few
examples of a given class (e.g. rare disease diagnosis, manufacturing defects, fradulent
transactions) which results in poor performance. Due to the nature of the data (occurrences
are so rare), it’s not always realistic to go out and acquire more. One way of solving this issue
is to under-sample the majority class. That is to say, we would exclude rows corresponding to
the majority class such that there are roughly the same amount of rows for both the majority
and minority classes. However, in doing so, we lose out on a lot of data that could be used to
train our model thus improving its accuracy (e.g. higher bias). Another other option is to
over-sample the minority class. In other words, we randomly duplicate observations of the
minority class. The problem with this approach is that it leads to overfitting because the
model learns from the same examples. This is where SMOTE comes in. At a high level, the
SMOTE algorithm can be described as follows:
● Take difference between a sample and its nearest neighbour
● Multiply the difference by a random number between 0 and 1
● Add this difference to the sample to generate a new synthetic example in feature space
● Continue on with next nearest neighbour up to user-defined number
Number of ‘bad’ values before applying SMOTE = 469
Number of ‘bad’ values after applying SMOTE = 65598

Future Scope:
1. Real-Time Fraud Detection Integration:
- Adapt regression models for real-time fraud detection, leveraging advanced anomaly
detection techniques and ensuring responsiveness to evolving fraudulent patterns.
2. Dynamic Credit Management Strategies:
- Develop dynamic credit management practices that adapt to changing economic
conditions and user behaviors, incorporating external factors for predicting credit limits in a
dynamic financial landscape.
3. User-Centric Security Measures:
- Integrate user-centric security measures, such as biometrics and behavioral analysis, to
complement regression models and provide an additional layer of protection.
4. Cross-Industry Applicability:
- Assess the applicability of regression models to other industries, fostering collaborations
to share insights and improve fraud prevention practices across diverse domains.
5. Blockchain for Enhanced Security:
- Explore the integration of blockchain or distributed ledger technologies to enhance the
security and transparency of credit card transactions, potentially reducing the risk of fraud
and unauthorized activities.

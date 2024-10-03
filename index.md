---
layout: default
---

## Background

Exoplanets are defined as planets that exist outside of our solar system. Some exoplanets have characteristics that make them more suitable for life than others. In the paper, “Habitability Classification of Exoplanets: A Machine Learning Insight”, Agrawal et. al. use various machine learning techniques to classify exoplanets from the PHL-EC dataset based on habitability to explore the limitations and suitability of different techniques. They found random forest, decision trees, and XGBoost performed the best in categorization. We want to explore this found insight using a dataset that categorized over 4000 unique exoplanets from the NASA Exoplanet Archive including 102 features, one of them being a boolean Habitality value.
Here is the dataset: [kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog](https://www.kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog)

## Problem Statement

Analysis on the data of these exoplanets is difficult and expensive as scientists have found thousands of exoplanets over the years with tons of data on them without a streamlined method to determine habitability on this data. Using PCA and KNN, we can aid in the determination of whether a given exoplanet is more likely to be habitable.


## Methods

Looking at the dataset, we noticed that there are many data points with missing values. To address this issue, we will first scan the entire dataset and delete the columns of features with a high percentage of missing values (>70%). With a large number of features (102) in this dataset, removing these missing columns should have a negligible impact on our results.

Due to the high dimensionality dataset, it may be beneficial to reduce dimensionality to focus on the most significant features and reduce overfitting. To achieve this, we can use Principal Component Analysis (PCA) to determine which principal components to keep based on correlations of the features. This should provide a simpler dataset which will likely improve performance and generalizability across the various ML models that we plan to implement.

Another data preprocessing method we can use is median imputation, where we replace missing values with the median of the feature, however, this method doesn’t guarantee an accurate prediction. An alternative approach using collaborative filtering is to implement the K-Nearest Neighbors algorithm to find similar instances and replace missing values based on the values of the nearest neighbors. This approach is more implementation heavy, and will be reserved for a smaller set of features with missing values after using other preprocessing methods. Additionally, many features have very different scales ranging from decimal differences to thousands. As a result, we will need to scale the data’s features through min-max scaling or z-score standardization to ensure optimal model performance.

Since habitability is a binary classification problem and the labels are provided in the dataset, we are working in a supervised learning setting which means we will need to segment our data into training and testing examples. It is important that we include enough cases of habitable planets in our each set due to the class imbalance.

As a baseline model, we can use logistic regression to classify the exoplanets as either habitable or non-habitable. Agrawal et. al. demonstrated the notable performance of tree-based methods such as decision trees, random forests, and gradient boosted trees on the PHL-EC dataset. Following this notion, we plan to implement all three tree-based algorithms, specifically using scikit-learn or XGBoost library for our ordinary decision tree and random forest models, and the XGBoost library for our gradient boosted model.

Similarly, support vector machines are effective in high dimensional spaces like our dataset, and this idea is accurately reflected by the strong SVM performance from Agrawal et. al. As such, we will implement the scikit-learn SVM model. 

Unlike the paper, we also plan to implement a deep neural network (multilayer perceptron) and compare its performance to the other models after sufficient hyperparameter tuning and regularization. Depending on how much control we want over the architecture of the network, we can choose to use PyTorch or streamline our implementation by using Keras.


## Expected Results

We expect that the model will have a high precision for identifying exoplanets that are potentially habitable and a recall rate that ensures minimal false positives. The PCA should help to increase model efficiency while maintaining prediction quality. In case the gaps in the dataset make it difficult to classify exoplanets as habitable or inhabitable, we will pivot to predicting a feature of the exoplanet considering if it is habitable or inhabitable.

## Metrics

For this project we will evaluate the performance of our model using several quantitative metrics. First, accuracy will measure how many exoplanets are correctly classified as habitable or not habitable. However, accuracy alone will not give us the full picture so we will also use precision, recall, and F1-score to better assess model performance. Precision will tell us how many exoplanets predicted to be habitable are actually habitable while recall will tell us how many of the exoplanets that are truly habitable have been correctly identified by the model. The F1-score will balance these metrics to make sure that false positives and negatives are minimized. 

## Project Goals

1.  In terms of accuracy, our goal is to achieve an accuracy of at least 80% when classifying habitability for the exoplanets. We chose 80% because that would give insights into how correct the model is without exceeding the scope of what is possible. Depending on how many features we use when classifying exoplanets as habitable, an 80% accuracy may be too high, but for now we want to aim for it.
2. In terms of precision, our goal is to achieve a precision of at least .7 when classifying habitability for the exoplanets. We want to ensure that there is high confidence when classifying a planet as habitable or inhabitable, but we understand that there can be false positives.
3. In terms of recall, our goal is to achieve a recall of at least .75 when classifying habitability for the exoplanets. We want to make sure the model correctly identifies a majority of habitable exoplanets and reduce the chances of missing habitable candidates. 
4. In terms of F1-Score, our goal is to achieve a F1-Score of at least .7 when classifying habitability for the exoplanets. This would keep balance between precision and recall but still minimize false positives and negatives.


## Contribution Table

| Name      | Contribution                                                |
|:----------|:------------------------------------------------------------|
| Varsha    |Results and Discussions: Expected result and project goals     |
| Harshitha |     |
| Muchen    |Data processing methods, github pages setup     |
| Sam       |Introduction/Background, Problem Statement     |
| Josh      |   |

## Gnatt Chart
```mermaid
gantt
    title Project Timeline
    dateFormat  YYYY-MM-DD
    section Section
    A task           :a1, 2014-01-01, 30d
    Another task     :after a1  , 20d
    section Another
    Task in sec      :2014-01-12  , 12d
    another task      : 24d
```
---
layout: default
---

## Background

Exoplanets are defined as planets that exist outside of our solar system. Some exoplanets have characteristics that make them more suitable for life than others. In the paper, “Habitability Classification of Exoplanets: A Machine Learning Insight”, Agrawal et. al. use various machine learning techniques to classify exoplanets from the PHL-EC dataset based on habitability to explore the limitations and suitability of different techniques. They found random forest, decision trees, and XGBoost performed the best in categorization. We want to explore this found insight using a dataset that categorized over 4000 unique exoplanets from the NASA Exoplanet Archive including 102 features, one of them being a boolean Habitality value.
Here is the dataset: [kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog](https://www.kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog)

## Problem Statement

Analysis on the data of these exoplanets is difficult and expensive as scientists have found thousands of exoplanets over the years with tons of data on them without a streamlined method to determine habitability on this data. Using PCA and KNN, we can aid in the determination of whether a given exoplanet is more likely to be habitable.

## Methods

Looking at the dataset, we’ve noticed that there are many data points with missing values. To address this issue, we will first scan the entire dataset and delete the columns of features with a high percentage of missing values (>70%), this would minimally affect the 102 features in the dataset.

Due to the high dimensionality dataset, it may be beneficial to reduce dimensionality to focus on the most significant features and reduce overfitting. To achieve this, we can use Principal Component Analysis (PCA) to determine which principal components to keep based on correlations of the features. This should provide a simpler dataset which will likely improve performance and generalizability across the various machine learning models that we plan to implement.

Another data preprocessing method we can use is median imputation, where we replace missing values with the median of the feature, however, this method doesn’t guarantee an accurate prediction. An alternative approach using collaborative filtering is to implement the KNN algorithm to find similar instances and replace missing values based on the values of the nearest neighbors. This approach is more implementation heavy, and will be reserved for a smaller set of features with missing values after using other preprocessing methods. Additionally, many features have very different scales ranging from decimal differences to thousands. Therefore, we will need to scale the data’s features through min-max scaling or z-score standardization to ensure optimal model performance.

Since habitability is a binary classification problem and the labels are provided in the dataset, we are working in a supervised learning setting which means we will need to segment our data into training and testing examples. It is important that we include enough cases of habitable planets in our each set due to the class imbalance.

Agrawal et. al. demonstrated the notable performance of tree-based methods such as decision trees, random forests, and gradient boosted trees on the PHL-EC dataset. Following this notion, we plan to implement all three tree-based algorithms, specifically using scikit-learn or XGBoost library for our ordinary decision tree and random forest models, and the XGBoost library for our gradient boosted model.

Similarly, support vector machines are effective in high dimensional spaces like our dataset, and this idea is accurately reflected by the strong SVM performance from Agrawal et. al. As such, we will implement the scikit-learn SVM model. 

Unlike the paper, we also plan to implement a deep neural network (multilayer perceptron) and compare its performance to the other models after sufficient hyperparameter tuning and regularization. Depending on how much control we want over the architecture of the network, we can choose to use PyTorch or streamline our implementation by using Keras.

## Expected Results

We expect that the model will have a high precision for identifying exoplanets that are habitable and a recall rate that ensures minimal false positives. The PCA should help to increase model efficiency while maintaining prediction quality. 

## Metrics

For this project we will evaluate the performance of our model using several quantitative metrics. First, accuracy will measure how many exoplanets are correctly classified as habitable or not habitable. However, accuracy alone will not give us the full picture so we will also use precision, recall, and F1-score to better assess model performance. Precision will tell us how many exoplanets predicted to be habitable are actually habitable while recall will tell us how many of the exoplanets that are truly habitable have been correctly identified by the model. The F1-score will balance these metrics to make sure that false positives and negatives are minimized. 

## Project Goals

1.  Accuracy: our goal is to achieve an accuracy of at least 80%. This will give insights into how correct the model is without exceeding the scope of what is possible. 
2. Precision: our goal is to achieve a precision of at least .7 We want to ensure that there is high confidence when classifying a planet, but we understand that there can be false positives.
3. Recall: our goal is to achieve a recall of at least .75. We want to make sure the model correctly identifies a majority of habitable exoplanets and reduce the chances of missing habitable candidates. 
4. F1-Score: our goal is to achieve a F1-Score of at least .7. This would keep balance between precision and recall but still minimize false positives and negatives.


## Works Cited

1. Basak, S., et al. "Habitability Classification of Exoplanets: A Machine Learning Insight." *The European Physical Journal Special Topics*, vol. 230, no. 9, 2021, pp. 2005-2023. *SpringerLink*, doi:10.1140/epjs/s11734-021-00203-z. Accessed 4 Oct. 2024.

2. "Importance of Feature Scaling." *Scikit-learn*, scikit-learn developers, https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html. Accessed 4 Oct. 2024.

3. "KNN." *Scikit-learn*, scikit-learn developers, https://scikit-learn.org/stable/modules/neighbors.html. Accessed 4 Oct. 2024.

4. "Neural Networks." *PyTorch Tutorials*, PyTorch, https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html. Accessed 4 Oct. 2024.

5. "PCA." *Scikit-learn*, scikit-learn developers, https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html. Accessed 4 Oct. 2024.

6. "Support Vector Machines." *Scikit-learn*, scikit-learn developers, https://scikit-learn.org/stable/modules/svm.html. Accessed 4 Oct. 2024.


## Contribution Table

| Name      | Contribution                                                |
|:----------|:------------------------------------------------------------|
| Varsha    |Results and Discussions: Expected result and project goals     |
| Harshitha |Results and Discussions: Expected Results and Metrics     |
| Muchen    |Data processing methods, github pages setup     |
| Sam       |Introduction/Background, Problem Statement     |
| Josh      |ML algorithm/model methods   |

## Gantt Chart
[Link to Gantt Chart.](./Exoplanet-Gantt-Chart.pdf)
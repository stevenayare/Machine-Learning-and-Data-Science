# Machine-Learning-and-Data-Science

Parkinson Prediction Logistic Regression Model
This repository contains a Parkinson Prediction Logistic Regression Model. The model is built using logistic regression to predict the presence or absence of Parkinson's disease based on various attributes.
Dataset
The dataset used in this model is the Parkinsons Data Set which is available from the UCI Machine Learning Repository. This dataset contains biomedical voice measurements from various people with early-stage Parkinson's disease, obtained from a telemonitoring device. The dataset has a total of 756 instances and 755 features.
Requirements
•	Python 3.6 or above
•	NumPy
•	Pandas
•	Seaborn
Model Training
The model was trained using logistic regression. The dataset was first loaded into a Pandas DataFrame and preprocessed. The preprocessing steps included dropping irrelevant columns, encoding categorical variables, and splitting the dataset into training and testing sets. The training set was used to train the model and the testing set was used to evaluate its performance.
After preprocessing, the model was trained using the training set. The trained model was then used to predict the presence or absence of Parkinson's disease in the testing set. The accuracy of the model was evaluated using various metrics such as confusion matrix, precision, recall, and F1-score.
Results
The model achieved an accuracy of 0.79, which means that it correctly predicted the presence or absence of Parkinson's disease in 79% of cases. The precision, recall, and F1-score were also calculated and reported in the code.
Usage
To use this model, you can simply download or clone the repository and run the Parkinsons.ipynb Jupyter notebook. The notebook contains all the code needed to preprocess the dataset, train the model, and evaluate its performance.
Conclusion
This Parkinson Prediction Logistic Regression Model provides a useful tool for predicting the presence or absence of Parkinson's disease based on various biomedical voice measurements. With an accuracy of 0.79, this model can be a valuable tool in diagnosing Parkinson's disease in its early stages.


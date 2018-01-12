#Machine learning algorithms used:
•	Support Vector Machine (SVM)
•	Reinforcement Neural Network (RNN)

#Datasets
Datasets taken for analyzing are of Apple Inc. (1980 – 2017) and Microsoft Corporation (1986 – 2017) from Equity-NASDAQ using Yahoo finance API. The data has been decomposed into two parts, 75% training dataset and 25% testing dataset.

Then comparison of the result was done with the testing dataset (i.e. 25% of the dataset). Based on the outcomes, the accuracy of both the algorithms were compared. For visualization of the accuracies of both the algorithms line graphs was used.


#Feature Selection
In the downloaded dataset 5 features were present for the daily stock price, namely:
Opening price
Closing price
High price
Low price
Volume


Increased the number of features to 11 from the 5 already present for better training of the model, namely:

Open change percentage list
Close change percentage list
High change percentage list
Low change percentage list
Volume change percentage list
Open difference percentage list
Volume difference percentage list
Open price moving average list
Close price moving average list
High price moving average list
Low price moving average list

Four feature lists were made comprising of all the 11 features to compare the accuracy for each set sets of features.

Feature list 1

Open change percentage list
Close change percentage list
High change percentage list
Low change percentage list
Volume change percentage list

Feature list 2

Open change percentage list
Close change percentage list
High change percentage list
Low change percentage list
Volume change percentage list
Open difference percentage list
Volume difference percentage list

Feature list 3

Open change percentage list
Close change percentage list
High change percentage list
Low change percentage list
Volume change percentage list
Open price moving average
Close price moving average
High price moving average
Low price moving average

Feature list 4

Open change percentage list
Close change percentage list
High change percentage list
Low change percentage list
Volume change percentage list
Open difference percentage list
Volume difference percentage list
Open price moving average
Close price moving average
High price moving average
Low price moving average


#Implementation
The implementation was done in Python environment with packages namely sci-kit learn, numpy, matplotlib and pybrain. The supervised learning models i.e. SVM (Support Vector Machine) and Recurrent Neural Network were executed and their accuracies obtained.
The models were implemented separately on both Apple and Microsoft stock prices of sets of 7 years’ data and 37 years’ data separately. 

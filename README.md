# EPL2022-Match-Prediction

## 1. EDA

Exploratory Data Analysis has been done in the following notebooks - 

1. [Analysis-v1 (Match Stats).ipynb](https://github.com/VIGNESHinZONE/EPL2022-Match-Prediction/blob/main/1.Analysis-v1%20(Match%20Stats).ipynb)
2. [Analysis-v2 (Team Performance).ipynb](https://github.com/VIGNESHinZONE/EPL2022-Match-Prediction/blob/main/2.%20Analysis-v2%20(Team%20Performance).ipynb)
3. [Analysis-v3 (Betting Odds).ipynb](https://github.com/VIGNESHinZONE/EPL2022-Match-Prediction/blob/main/3.%20Analysis-v3%20Betting%20Odds.ipynb)

Go through the notebooks in the above mentioned order

## 2. ML Modelling

The Notebook for ML Modelling is done in [Modelling.ipynb](https://github.com/VIGNESHinZONE/EPL2022-Match-Prediction/blob/main/Modelling.ipynb).

After having thoroughly explored the dataset and building multiple simple baseline models using the Probability scores or Point Difference or Form Difference, we were able to achieve an approx 72%. Any ML we perform should perform better than our Baseline models. We will be exploring different ML algorithms to build this strategy.


### 2.1 Important Features

Upon performing EDA, we have identified a few essential features that could determine the game's outcome. They are listed below - 

1. Winning Probabilities from different Betting Houses - 
    - Just on building a simple comparison model, we were able to generate 72% accuracy.
    - We will also look at the difference between Opening and Closing Bets of various houses.
    
2. Point difference
    - It measures the difference in point gained by the teams through the season. The baseline created using point difference gave us a 70% accuracy.
    
3. Form Difference
    - We have built a custom formula to measure the current form of a team, which was discussed in `2. Analysis-v2 (Team Performance).ipynb`. Just using this formula, we were able to achieve 70 to 72% accuracy.
    
4. Elite Teams
    - Let's create a unique feature to indicate that a Team is Elite. It has been observed that Betting Probabilities have a hard time predicting when Elite Teams are playing away. Elite Teams are - {'Man United', 'Chelsea', 'Tottenham', 'Liverpool', 'Man City', 'Arsenal'}
    
5. New Teams of Season
    - Every season, the bottom three teams get relegated to a lower division, and three new teams from the lower division are included. These teams tend to be weaker playing sides. New Teams are - {'Norwich', 'Watford', 'Brentford'}


### 2.2 Cross-Validation

It was observed that our Baseline model's performance was a bit weaker in the half of the tournament. So to effectively model our system, we adopted two validation strategies -

1. We train a model on the first 30 weeks of the dataset and validate its performance on the last 8 weeks.
2. We Randomly sample 80% of our dataset for the Train set and the rest of the 20% for validation. We might also experiment with K-fold random splitting as well

### 2.3 Evaluation

For the task, we need to successfully predict if the Away team has won the game or not. Since we do not know how detrimental a False Positive or False Negative could be, both Recall & Precision are essential. Here is the list of metrics we will note to evaluate the performance of our ML model - 

1. Recall
2. Precision
3. Accuracy
4. ROC AUC score
5. Confusion Matrix

### 2.4 Features Neglected or will be Later explored

1. We have decided not to use the Asian Handicap Betting Odds as there seems to be a slight confusion. The resources I had read online mentioned that a small penalty value gets added to the final score. But the key value descriptions do not provide information about the type of Asian handicap being used. Hence, I have decided to move forward with things that I already know.

2. We would like to later explore the effect of Total Goals and its betting odds on the outcome of the game.

3. We would like to explore the effect of information about the Day of the week and the week when the game was played.

### 2.5 Features Neglected or will be Later explored

Prior to Building a Machine Learning Model, we had built a simple hueristic model which was assigned to determine if Away team would win the match or not by comparing the Odds, Table Points and Current Form.

We had achieved the following accuracy on Week Based Validation - 

| Feature Classifier      | Accuracy | Precison | Recall | 
| :---------------------: | :------: | :------: | :----: |
| Opening Betting Odds    | 0.695    | 0.516    | 0.615  |
| Closing Betting Odds    | 0.720    | 0.552    | 0.615  |
| Point Difference        | 0.671    | 0.486    | 0.692  |
| Form Difference 1       | 0.707    | 0.538    | 0.538  |
| Form Difference 2       | 0.671    | 0.486    | 0.654  |

Following metrics were achieved on the k-fold validation dataset - 

| Feature Classifier      | Accuracy | Precison | Recall | 
| :---------------------: | :------: | :------: | :----: |
| Opening Betting Odds    | 0.721    | 0.579    | 0.664  |
| Closing Betting Odds    | 0.743    | 0.610    | 0.680  |
| Point Difference        | 0.680    | 0.536    | 0.520  |
| Form Difference 1       | 0.680    | 0.535    | 0.496  |
| Form Difference 2       | 0.628    | 0.468    | 0.632  |


Using the closing Betting Odds, we managed to achieve an accuracy of 72%.

Now let us go over our ML Modelling stratergies. Upon exploring multiple classification models, Logistic Regression & XGBoost were the ones which gave us best results. Along with this, we had also added SMOTE Sampling in some our experiments which improved the recall of our models. Later we even built a ensemble (Stacking) model combining the predtive powers of Logistic Regression and XGBoost. Listed below are the accuracies of our best performing experiments - 

| Model Description      | SMOTE | Cross Validation Dataset | Accuracy | Precison | Recall | AUC Score |
| :---------------------:| :---: | :----------------------: | :------: | :------: | :----: | :-------: |
| Logistic Regresion L1  | No    | Week Based Dataset       | 0.707    | 0.536    | 0.577  | 0.689     |
| Logistic Regresion L1  | No    | K-fold Dataset           | 0.713    | 0.651    | 0.360  | 0.687     |
| XGBoost                | Yes   | Week Based Dataset       | 0.732    | 0.562    | 0.692  | 0.714     |
| XGBoost                | No    | K-fold Dataset           | 0.680    | 0.548    | 0.464  | 0.715     |
| Stacking               | Yes   | Week Based Dataset       | 0.707    | 0.529    | 0.692  | 0.716     |
| Stacking               | No    | K-fold Dataset           | 0.683    | 0.549    | 0.480  | 0.712     |

## 3. Conclusion

Let's compare the performance of Hueristic models to ML Models from the table presented above. ML Models seem to perform better in Week Based Dataset, and Hueristic Models seem to serve better in Sampled (Stratified) k-fold dataset. We achieved an accuracy of 73% (56% precision and 70% recall) on ML Models trained on week based datasets. And we achieved an accuracy of 74% (61% precision and 68% recall) built on the k-fold dataset using heuristics. 

Depending on the application of the problem, we can choose the appropriate modelling strategy. The ML model might be more suitable in a scenario where we want to place bets after collecting information about the teams' performance in the season. Similarly, relying on betting odds might be more suitable if we were to predict the outcome of any random game.

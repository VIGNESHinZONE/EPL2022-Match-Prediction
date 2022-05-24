# EPL2022-Match-Prediction

### EDA

Exploratory Data Analysis has been done in the following notebooks - 

1. Analysis-v1 (Match Stats).ipynb
2. Analysis-v2 (Team Performance).ipynb
3. Analysis-v3 (Betting Odds).ipynb

Go through the notebooks in the above mentioned order

### ML Modelling

After having thoroughly explored the dataset and building multiple simple baseline models using the Probability scores or Point Difference or Form Difference, we were able to achieve an approx 72%. Any ML we perform should perform better than our Baseline models. We will be exploring different ML algorithms to build this strategy.


#### Important Features

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


#### Cross-Validation

It was observed that our Baseline model's performance was a bit weaker in the half of the tournament. So to effectively model our system, we adopted two validation strategies -

1. We train a model on the first 30 weeks of the dataset and validate its performance on the last 8 weeks.
2. We Randomly sample 80% of our dataset for the Train set and the rest of the 20% for validation. We might also experiment with K-fold random splitting as well

#### Evaluation

For the task, we need to successfully predict if the Away team has won the game or not. Since we do not know how detrimental a False Positive or False Negative could be, both Recall & Precision are essential. Here is the list of metrics we will note to evaluate the performance of our ML model - 

1. Recall
2. Precision
3. Accuracy
4. ROC AUC score
5. Confusion Matrix

## Features Neglected or will be Later explored

1. We have decided not to use the Asian Handicap Betting Odds as there seems to be a slight confusion. The resources I had read online mentioned that a small penalty value gets added to the final score. But the key value descriptions do not provide information about the type of Asian handicap being used. Hence, I have decided to move forward with things that I already know.

2. We would like to later explore the effect of Total Goals and its betting odds on the outcome of the game.

3. We would like to explore the effect of information about the Day of the week and the week when the game was played.

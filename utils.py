import pandas as pd
import numpy as np
import math

def build_ongoing_points_table(df, team_names, elite_teams):
    """
    1. Custom Form formula 1- 
        {
        +1 for win,
        +1 if opponent team is elite,
        +1 if its away,
        +0 if draw,
        -1 for defeat
        }

    2. Custom Form formula - 
        {
        Performance = {
                        if team of interest is away - 
                            -1.2 * (Home Goal - Away Goal)
                        if team of interest is home - 
                            +1 * (Home Goal - Away Goal)
                      }
        point difference = {Points of Team of Interest - Points of Opponent}

        point difference weightage = max(1, 1/(1+ln(point difference)))

        final score = Performance * point difference weightage
        }
    """
    Teams = df['HomeTeam'].unique().tolist()
    curr_team_points = {team: 0 for team in team_names}
    curr_team_form1 = {team: 1 for team in team_names}
    curr_team_form2 = {team: 1 for team in team_names}
    home_point, away_point = [], []
    home_form1, away_form1 = [], []
    home_form2, away_form2 = [], []
    for index, row in df.iterrows():
        home_point.append(curr_team_points[row['HomeTeam']])
        away_point.append(curr_team_points[row['AwayTeam']])
        home_form1.append(curr_team_form1[row['HomeTeam']])
        away_form1.append(curr_team_form1[row['AwayTeam']])
        home_form2.append(curr_team_form2[row['HomeTeam']])
        away_form2.append(curr_team_form2[row['AwayTeam']])

        home_score1 = 0
        away_score1 = 0.5
        home_score2 = (row['FTHG'] - row['FTAG']) 
        away_score2 = -1.2*home_score2
        away_score2 = max(-1.5, away_score2)
        
        home_point2 = curr_team_points[row['HomeTeam']] 
        away_point2 = curr_team_points[row['AwayTeam']] 
        point_diff_home = home_point2 - away_point2
        point_diff_away = -1 * point_diff_home 
        point_diff_home = max(0.001, point_diff_home)
        point_diff_away = max(0.001, point_diff_away)
        home_score2 *= (math.log(home_point2+3)/(1+math.log(point_diff_home+1, 10)))
        away_score2 *= (math.log(away_point2+3)/(1+math.log(point_diff_away+1, 10)))
        if row['FTR'] == 'H':
            curr_team_points[row['HomeTeam']] += 3
            home_score1 += 1
            away_score1 -= 1
        elif row['FTR'] == 'A':
            curr_team_points[row['AwayTeam']] += 3
            away_score1 += 1
            home_score1 -= 1
        else:
            curr_team_points[row['HomeTeam']] += 1
            curr_team_points[row['AwayTeam']] += 1
        
        if row['HomeTeam'] in elite_teams:
            away_score1 += 0.5
        if row['AwayTeam'] in elite_teams:
            home_score1 += 0.5
        
        curr_team_form1[row['HomeTeam']] *= 0.75
        curr_team_form1[row['HomeTeam']] += home_score1
        curr_team_form1[row['AwayTeam']] *= 0.75
        curr_team_form1[row['AwayTeam']] += away_score1

        curr_team_form2[row['HomeTeam']] *= 0.75
        curr_team_form2[row['HomeTeam']] += home_score2
        curr_team_form2[row['AwayTeam']] *= 0.75
        curr_team_form2[row['AwayTeam']] += away_score2
    
    df['HomePoint'] = home_point
    df['AwayPoint'] = away_point
    df['HomeForm1'] = home_form1
    df['AwayForm1'] = away_form1
    df['HomeForm2'] = home_form2
    df['AwayForm2'] = away_form2
    return df

def find_optimal_threshold(df, feature_name):
    mint, maxt = df[feature_name].min(), df[feature_name].max()
    thres = np.linspace(int(mint), int(maxt), num=int(maxt) - int(mint)+1)
    max_score = 0
    max_t = None
    for t in thres:
        score_t = ((df[feature_name] > t).astype(int) == df['Result']).mean()
        if score_t > max_score:
            max_score = score_t
            max_t = t
    
    print(f"For {feature_name} at threshold {max_t}, we get accuracy {max_score}")
    return max_score, max_t

def generate_betting_probablity(df, betting_houses, betting_houses_closing):
    for houses in betting_houses:
        df[houses+'_total_prob'] = 0.0
        for keys in betting_houses[houses]:
            df[keys + '_false_prob'] = df[keys].apply(lambda x: 1/x)
            df[houses+'_total_prob'] += df[keys + '_false_prob']

        for keys in betting_houses[houses]:
            df[keys + ' Prob'] = df[keys + '_false_prob'] / df[houses+'_total_prob']

        df.drop([keys + '_false_prob' for keys in betting_houses[houses]] + [houses+'_total_prob'], axis=1)

    for houses in betting_houses_closing:
        df[houses+'_total_prob'] = 0.0
        for keys in betting_houses_closing[houses]:
            df[keys + '_false_prob'] = df[keys].apply(lambda x: 1/x)
            df[houses+'_total_prob'] += df[keys + '_false_prob']

        for keys in betting_houses_closing[houses]:
            df[keys + ' Prob'] = df[keys + '_false_prob'] / df[houses+'_total_prob']

        df.drop([keys + '_false_prob' for keys in betting_houses_closing[houses]] + [houses+'_total_prob'], axis=1)

    features_names = [houses + ' Away Prob Diff' for houses in betting_houses]
    for houses in betting_houses:
        df[houses + ' Away Prob Diff'] = df[betting_houses_closing[houses][2] + ' Prob'] - df[betting_houses[houses][2] + ' Prob']
    
    return df
#Project

I chose to do the prediction on soccer data because I am a big fan of soccer. I love to play and watch soccer especially from England Premier League. There are currently 20 teams in the league, and each team has played 9 to 11 games in the season. England Premier League is one of the most competitive leagues in the world, and it is fun to predict which team will win the title. For this project, I want to predict, based on the current stats, which teams will be the top 4 teams in 2020-21 season because those teams get to play in the world’s biggest club competition called Champions League. 

I collected data from 2014-15 season up to 2020-21 season (2020-21 is the current season). I wanted to scrap data for a few more years, but the data before 2014-15 season only had stats partially. With those data, I made two sets of data: one has the data just for the current season, and the other one has the data for the rest of the seasons. I used the latter data to make a prediction model and used that model to get the top 4 teams based on the current stats. The data has 9 variables: team, points, goals, goals allowed, shots on target, possession, fouls, saves by goalkeeper, and age. I set y variable with points and X variable with the rest except team.
Feature definitions are as following


Points: total points earned in a season (win=3 points, draw=1 point, loss=0 point)

Goals: total number of goals made in a season

Goals allowed: total number of goals conceded in a season 

Shots on target: total number of shot attempts that would or does enter the goal if left unblocked in a season

Possession: the average percentage a team possesses the ball during a game in a season

Fouls: total number of fouls committed in a season

Saves by goalkeeper: total number of saves made by goalkeeper in a season

Age: the average age of players in a team

I explored 4 models: linear regression, Lasso regression, KNN model, and Decision tree model. I chose these models because I was most familiar with these 4 models. Then, I performed a cross validation to find the best model and discovered that linear regression and Lasso regression have lower cross validation score than the other 2. I chose Lasso regression because it has the lowest score with 19.43. After choosing the model, I fitted the model with the data from 2014-15 to 2019-2020 season and made a prediction based on the current stats of the teams. However, I felt that the prediction was quite not there yet, so I added a feature of goals allowed, and that decently changed the prediction.

As a result, I found the top 4 teams, and those are Chelsea, Tottenham, Liverpool, and Aston Villa in order. Based on my model, Chelsea was predicted to be the champion of the 2020-21 season of England Premier League with approximately 65 points. The predicted points are smaller than the usual winning points because the current season is still going on, and the model used only partial stats. I also went on to get important features, and the features of goals and goals allowed were the most impactful features in the model. The figure in the markdown file shows that more goals can increase the points, and more goals allowed can decrease the points. (The figure is in the markdown file. I tried to get it here, but didn't work)

Feature: [0] Goals Score: 0.66

Feature: [1] Goals_Allowed Score: -0.62

I want to conclude my report with some existing limitations. My X variable which contained the data from 2014-15 to 2019-20 season only had 6 seasons of data, and, in addition, I was only able to get 7 features. The prediction would have been better and more accurate if I had more rows of data and features in the model.  



















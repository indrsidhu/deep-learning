import numpy as np # numerical computing 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization
import seaborn as sns #modern visualization

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

matches = pd.read_csv('matches.csv')

'''
Question: How many matches we’ve got in the dataset?
'''
print(matches['id'].max())

'''
Question: How many seasons we’ve got in the dataset?
'''
print(matches['season'].unique())
print(len(matches['season'].unique()))

'''
Question : Which Team had won by maximum runs?
'''
print(matches.iloc[matches['win_by_runs'].idxmax()])

'''
Quesion: Which Team had won by maximum wickets?
'''
print(matches.iloc[matches[matches['win_by_runs'].ge(1)].win_by_runs.idxmin()]['winner'])

'''
Quesion: Which Team had won by minimum wickets?
'''
print(matches.iloc[matches[matches['win_by_wickets'].ge(1)].win_by_wickets.idxmin()])

'''
Quesion: Which season had most number of matches?
'''
sns.countplot(x='season', data=matches)
plt.show()


'''
Question: The most successful IPL Team
'''
data = matches.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h');

plt.show()
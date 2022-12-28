import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
print(df.head())
print(df.describe().T)
df.columns = df.columns.str.lower()

# perform exploratory analysis here:
#plt.scatter(df.wins,df.breakpointsfaced)
#plt.show()
#plt.clf()
#plt.scatter(df.wins,df.winnings)
#plt.show()

## perform single feature linear regressions here:
features = df[['breakpointsfaced']]
outcome = df[['wins']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features,outcome,train_size=0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction,alpha=0.4)
plt.title('breakpointsfaced vs wins')
plt.show()



















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:

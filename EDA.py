import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# training data exploration
train_df = pd.read_csv("data\Train.csv")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(train_df.head())

# explore location
percentage_NA_location = sum(pd.isnull(train_df['location'])) / len(train_df['location'])
print(f'Percentage of missing location: {percentage_NA_location}')

# explore sample
sample_distribution = train_df['target'].value_counts()
print(sample_distribution)

print(sample_distribution[1] / (sample_distribution[0] + sample_distribution[1]))

sns.countplot(train_df['target'])
plt.show()

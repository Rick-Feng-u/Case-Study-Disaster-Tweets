import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# training data exporeation
train_df = pd.read_csv("data\Train.csv")
sample_distribution = train_df['target'].value_counts()
print(sample_distribution)

print(sample_distribution[1] / (sample_distribution[0] + sample_distribution[1]))

sns.countplot(train_df['target'])
plt.show()


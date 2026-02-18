# Unsupervised Learning

[Unsupervised Learning](https://www.codecademy.com/articles/machine-learning-supervised-vs-unsupervised) is a type of [machine learning](https://www.codecademy.com/resources/docs/general/machine-learning) where the program learns the inherent structure of the data based on unlabeled examples.

**Clustering** is a common unsupervised machine learning approach that finds patterns and structures in unlabeled data by grouping them into clusters.

Some examples:

- Social networks clustering topics in their news feed
- Consumer sites clustering users for recommendations
- Search engines to group similar objects in one cluster

For a quick preview, here’s an example of unsupervised learning:

A social media platform wants to separate their users into categories based on what kind of content they engage with. They have collected three pieces of data from a sample of users:

- Number of hours per week spent reading posts
- Number of hours per week spent watching videos
- Number of hours per week spent in virtual reality

The company is using an [algorithm](https://www.codecademy.com/resources/docs/general/algorithm) called k-means clustering to sort users into three different groups.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import codecademylib3
from plot import plot_clusters

# Load the data
media_usage = pd.read_csv('media_usage.csv')

# Create the model
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(media_usage)

labels = kmeans.predict(media_usage)

# Plot the clusters
plot_clusters(media_usage, labels)
```
# Supervised Learning

## Regression

Machine Learning can be branched out into the following categories:

- Supervised Learning
- Unsupervised Learning

[Supervised Learning](https://www.codecademy.com/articles/machine-learning-supervised-vs-unsupervised) is where the data is labeled and the program learns to predict the output from the input data. For instance, a supervised learning [algorithm](https://www.codecademy.com/resources/docs/general/algorithm) for credit card fraud detection would take as input a set of recorded transactions. For each [transaction](https://www.codecademy.com/resources/docs/general/database/transaction), the program would predict if it is fraudulent or not.

Supervised learning problems can be further grouped into regression and classification problems.

**Regression:**

In regression problems, we are trying to predict a continuous-valued output. Examples are:

- What is the housing price in New York?
- What is the value of cryptocurrencies?

**Classification:**

In classification problems, we are trying to predict a discrete number of values. Examples are:

- Is this a picture of a human or a picture of a cyborg?
- Is this email spam?

## Classification Example

An exclusive nightclub in Neo York doesn’t want to serve robots, but technology has advanced so far that it’s hard for bouncers to tell humans from robots just by looking. To help the bouncers, the nightclub created a model that uses the k-nearest neighbors [algorithm](https://www.codecademy.com/resources/docs/general/algorithm) to distinguish between humans and robots based on how long it takes them identify blurry pictures or traffic lights.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the data
photo_id_times = pd.read_csv('photo_id_times.csv')

# Separate the data into independent and dependent variables
X = np.array(photo_id_times['Time to id photo']).reshape(-1, 1)
y = photo_id_times['Class']

# Create a model and fit it to the data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

time_to_identify_picture = 5

# Make a prediction based on how long it takes to identify a picture
y_pred = neigh.predict(np.array(time_to_identify_picture).reshape(1, -1))

if y_pred == 1:
    print("We think you're a robot.")
else:
    print("Welcome, human!")
```

---

When we explicitly tell a program what we expect the output to be, and let it learn the rules that produce expected outputs from given inputs, we are performing supervised learning.

![supervised-learning.gif](supervised-learning.gif)
#  Transforming Data into Features
You are a data scientist at a clothing company and are working with a data set of customer reviews. This dataset is originally from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) and has a lot of potential for various machine learning purposes. You are tasked with transforming some of these features to make the data more useful for analysis. 

To do this, you will have time to practice the following:
- Transforming categorical data
- Scaling your data
- Working with date-time features

---

## Original Dataset (Raw Features)

| clothing_id | age | review_date | rating     |
|-------------|-----|------------|------------|
| 1095        | 39  | 2019-07-08 | Liked it   |
| 1095        | 28  | 2019-05-17 | Loved it   |
| 699         | 37  | 2019-06-24 | Loved it   |
| 1072        | 36  | 2019-12-06 | Loved it   |
| 1094        | 32  | 2019-10-04 | Loved it   |
| ...         | ... | ...        | ...        |
| 918         | 38  | 2019-05-26 | Loved it   |
| 950         | 33  | 2019-10-21 | Hated it   |
| 1086        | 36  | 2019-10-18 | Loved it   |
| 1033        | 28  | 2019-11-24 | Loved it   |
| 850         | 64  | 2019-10-31 | Loved it   |

## Transformed Dataset (Feature-Engineered)

| clothing_id | age | recommended | rating | Bottoms | Intimate | Jackets | Tops | Trend |
|-------------|-----|-------------|--------|----------|-----------|----------|-------|--------|
| 1095        | 39  | 1           | 4      | 0        | 0         | 0        | 0     | 0      |
| 1095        | 28  | 1           | 5      | 0        | 0         | 0        | 0     | 0      |
| 699         | 37  | 1           | 5      | 0        | 1         | 0        | 0     | 0      |
| 1072        | 36  | 1           | 5      | 0        | 0         | 0        | 0     | 0      |
| 1094        | 32  | 1           | 5      | 0        | 0         | 0        | 0     | 0      |
| ...         | ... | ...         | ...    | ...      | ...       | ...      | ...   | ...    |
| 918         | 38  | 1           | 5      | 0        | 0         | 0        | 1     | 0      |
| 950         | 33  | 0           | 1      | 0        | 0         | 0        | 1     | 0      |
| 1086        | 36  | 1           | 5      | 0        | 0         | 0        | 0     | 0      |
| 1033        | 28  | 1           | 5      | 1        | 0         | 0        | 0     | 0      |
| 850         | 64  | 1           | 5      | 0        | 0         | 0        | 1     | 0      |
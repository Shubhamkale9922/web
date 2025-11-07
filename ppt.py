Experiment 1: Logistic Regression on Iris Dataset

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


Experiment 2: Decision Tree on Titanic Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()


Experiment 3: Naive Bayes Sentiment Analysis

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

reviews = [
    "I love this movie", "This film was great", "Amazing storyline",
    "I hated this movie", "The film was terrible", "Worst movie ever"
]
labels = [1, 1, 1, 0, 0, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


Experiment 4: Simple Linear Regression (Salary Prediction)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [35000, 37000, 40000, 43000, 45000, 48000, 50000, 52000, 55000, 58000]
})

X = data[['Experience']]
y = data['Salary']

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.show()

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)


Experiment 5: Linear Regression (House Price Prediction)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sqft = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
price = np.array([150000, 200000, 250000, 280000, 350000])

model = LinearRegression()
model.fit(sqft, price)
pred = model.predict(sqft)

mse = mean_squared_error(price, pred)
r2 = r2_score(price, pred)

print("MSE:", mse)
print("R² Score:", r2)


Experiment 6: K-Means Clustering

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=100, centers=7, random_state=42)

kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("K-Means Clustering with K=7")
plt.show()


Experiment 7: Frequent Pattern Tree (FP-Growth)

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

dataset = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs']
]

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_items = fpgrowth(df, min_support=0.4, use_colnames=True)
print(frequent_items)


Experiment 8: Apriori Algorithm

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs']
]

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_items = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

print("Frequent Itemsets:\n", frequent_items)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence']])
MCA Data Science Lab Experiments (9 to 12)
Experiment 9: FP-Growth Algorithm for Market Basket Analysis

# Experiment 9: FP-Growth Market Basket Analysis
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Sample retail transactions
dataset = [
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'diapers', 'beer', 'bread'],
    ['milk', 'diapers', 'bread', 'butter'],
    ['bread', 'butter']
]

# One-hot encode the dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply FP-Growth
frequent_items = fpgrowth(df, min_support=0.4, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

# Show top 5 rules sorted by confidence
rules = rules.sort_values(by="confidence", ascending=False).head(5)
print("Top 5 Association Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence']])
Experiment 10: Create Bar Chart (Stacked)

# Experiment 10: Stacked Bar Chart (Revenue by Month & Region)
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    'Month': ['Mar', 'Apr', 'May', 'Jun', 'Jul'],
    'East': [100, 120, 90, 110, 115],
    'West': [80, 95, 70, 100, 90],
    'North': [60, 70, 80, 85, 95]
}

df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

# Plot stacked bar chart
df.plot(kind='bar', stacked=True, color=['green', 'orange', 'brown'])
plt.title("Monthly Revenue by Region")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend(title="Region")
plt.show()
Experiment 11: Bar Chart of critics_score

# Experiment 11: Bar Chart for Movie Critics' Scores
import pandas as pd
import matplotlib.pyplot as plt

# Create a small sample movies dataset (replace this with reading CSV)
data = {
    'movie_title': ['Movie1','Movie2','Movie3','Movie4','Movie5','Movie6','Movie7','Movie8','Movie9','Movie10'],
    'critics_score': [85, 78, 92, 88, 70, 65, 95, 80, 82, 90]
}
df = pd.DataFrame(data)

# Plot bar chart
plt.figure(figsize=(8, 4))
plt.bar(df['movie_title'], df['critics_score'], color='skyblue')
plt.title("Critics Score for First 10 Movies")
plt.xlabel("Movie")
plt.ylabel("Critics Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("movie_scores_plot.png")
plt.show()

print("Plot saved as 'movie_scores_plot.png'")
Experiment 12: Boxplot for mpg and cyl using mtcars dataset

# Experiment 12: Boxplot for mpg and cyl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load mtcars dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv"
df = pd.read_csv(url)

# Create boxplot
sns.boxplot(x='cyl', y='mpg', data=df, palette='pastel')
plt.title("Boxplot of MPG by Number of Cylinders")
plt.xlabel("Cylinders")
plt.ylabel("Miles Per Gallon (MPG)")
plt.show()

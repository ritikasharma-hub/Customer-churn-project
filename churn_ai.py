import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

df = pd.read_csv("data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.info())


sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()


#data cleaning and preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

#drop id column
df.drop('customerID', axis=1, inplace=True)

#encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

#one hot enndcoding
df = pd.get_dummies(df, drop_first=True)

#feature engineering
df['Charges_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

#train test split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#feature selection
mi_scores = mutual_info_classif(X_train, y_train)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

print(mi_df.head(10))


# logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
print("Logistic Regression")
print(classification_report(y_test, lr_pred))

#tree based(random forest)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest")
print(classification_report(y_test, rf_pred))



#neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

print("Neural Network")
print(classification_report(y_test, mlp_pred))


#hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1'
)

grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)


#eerror analysis
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix – Random Forest")
plt.show()


#unstrucrured data
text_df = pd.read_csv("data\Tweets.csv")
text_df = text_df[['text', 'airline_sentiment']].dropna()

#sentiment label encoding
text_df['sentiment'] = text_df['airline_sentiment'].map({
    'negative': 0,
    'neutral': 1,
    'positive': 1
})


#text preprocessing and vectorisation
stop_words = stopwords.words('english')

tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words=stop_words
)

X_text = tfidf.fit_transform(text_df['text'])
y_text = text_df['sentiment']


#text classification model
text_model = LogisticRegression(max_iter=500)
text_model.fit(X_text, y_text)

text_df['sentiment_score'] = text_model.predict_proba(X_text)[:, 1]


# integration text insight
avg_sentiment = text_df['sentiment_score'].mean()
df['sentiment_score'] = avg_sentiment


#return churn model with sentiment
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf.fit(X_train, y_train)
final_pred = rf.predict(X_test)

print("Final Model with Sentiment")
print(classification_report(y_test, final_pred))


















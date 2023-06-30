import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

titanic_data = pd.read_csv('C:\\Users\\Harsh\\OneDrive\\Desktop\\Bharat Intern\\Titanic_dataset.csv')

selected_features = ['Pclass', 'Age', 'Sex']
features = titanic_data[selected_features]
target = titanic_data['Survived']

missing_target_indices = target.isnull()
features = features[~missing_target_indices]
target = target[~missing_target_indices]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

numerical_features = ['Age']
categorical_features = ['Pclass', 'Sex']

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(features_train, target_train)

target_pred = pipeline.predict(features_test)

accuracy = accuracy_score(target_test, target_pred)
print("Accuracy:", accuracy)

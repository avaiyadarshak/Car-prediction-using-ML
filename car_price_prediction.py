import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib

# load dataset
df = pd.read_csv("cars.csv")

# display dataset
df.head()
df.info()
df.describe()

# dataset shape
df.shape

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)
target = "Ex-Showroom_Price"
df[target] = df[target].astype(str)
df[target] = df[target].str.replace(",", "")
df[target] = df[target].str.replace("Rs.", "")
df[target] = df[target].str.strip()
df[target] = pd.to_numeric(df[target], errors="coerce")
df.dropna(subset=[target], inplace=True)

X = df.drop(target, axis=1)
y = df[target]

num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

numeric = Pipeline(
[
("imputer",SimpleImputer(strategy="median"))
]
)

categorical = Pipeline(
[
("imputer",SimpleImputer(strategy="most_frequent")),
("encoder",OneHotEncoder(handle_unknown="ignore"))
]
)


preprocessor = ColumnTransformer(
[
("num",numeric,num_cols),
("cat",categorical,cat_cols)
]
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model = Pipeline([
    ("preprocessor", preprocessor),

    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R2 Score :", r2)

result = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

result.head(10)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Price")

plt.show()


import joblib

joblib.dump(model, "car_price_model.pkl")

print("Model Saved Successfully")

model = joblib.load("car_price_model.pkl")

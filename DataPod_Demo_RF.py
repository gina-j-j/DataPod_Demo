# %% [markdown]
# ## DATA

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("DataPod.csv")
df

# %%
df = df[["DATE", "DISSOLVED OXYGEN", "ORP", "PH", "CONDUCTIVITY","TEMPERATURE"]].copy()

# %%
df["DATE"] = pd.to_datetime(df["DATE"])
df.set_index("DATE", inplace=True)
df = df.sort_index(ascending=True)

# %%
df.columns = ["DO", "ORP", "PH", "COND", "TEMP"]

# %%
df['bloom'] = ((df['PH'] > 8.5) & (df['DO'] > 11) & (df['ORP'] < 400)).astype(int)

# %%
df['target'] = df['bloom'].shift(-1)

# %%
df = df.dropna()

# %%
X = df[["DO", "ORP", "PH", "COND", "TEMP"]]
y = df['target']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %% [markdown]
# ## TRAINING

# %% [markdown]
# ### RF

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,f1_score, make_scorer
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# %%
def objective(trial):
    rf_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
    }

    rf_model = RandomForestClassifier(**rf_params)
    f1_scorer = make_scorer(f1_score, zero_division=0)

    rf_model_scores = cross_val_score(
        rf_model,
        X_train,
        y_train,
        cv=tscv,
        scoring=f1_scorer
    )

    mean_score = rf_model_scores.mean()
    return mean_score if not np.isnan(mean_score) else 0.

# %%
rf_study=optuna.create_study(direction="maximize")
rf_study.optimize(objective, n_trials=20)

# %%
rf_best_params = rf_study.best_params
print(rf_best_params)
rf = RandomForestClassifier(**rf_best_params)
rf.fit(X_train, y_train)

# %% [markdown]
# ## TESTING

# %%
predictions = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nFeature Importance:")
importance = pd.Series(rf.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))

# %%
results = X_test.copy()

results['Bloom_Actual'] = y_test
results['Model_Prediction'] = predictions

probabilities = rf.predict_proba(X_test)[:, 1]
results['Bloom_Probability_%'] = (probabilities * 100).round(2)

def risk_level(prob):
    if prob > 80: return "HIGH"
    if prob > 40: return "MEDIUM"
    return "LOW"

results['Risk_Level'] = results['Bloom_Probability_%'].apply(risk_level)
results['Bloom_Probability_%'] = results['Bloom_Probability_%'].astype(str) + '%'

results['Correct'] = results['Bloom_Actual'] == results['Model_Prediction']

results = results.sort_index()

cols_to_print = ["DO", "ORP", "PH", "COND", "TEMP", 'Model_Prediction', 'Correct', 'Bloom_Probability_%', 'Risk_Level']
print(results[cols_to_print].head(15))

print("\nAccuracy Count:")
print(results['Correct'].value_counts())

# %%
import matplotlib.pyplot as plt
df_oy = df.loc['2025-07-30':'2025-08-30']
plt.style.use('dark_background')
plt.figure(figsize=(20,6))
plt.plot(df_oy['DO'], color='cornflowerblue')
plt.xlabel('Date')
plt.ylabel('DO Levels')
plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.6)
plt.legend()
plt.show()

# %%




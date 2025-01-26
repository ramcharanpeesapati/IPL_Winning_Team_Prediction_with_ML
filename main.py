import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\\Users\\lrgud\\OneDrive\\Desktop\\IPL_Winning_Team_Prediction\\ipl_colab.csv"
data = pd.read_csv(file_path)

def feature_engineering(data):
    data = data[['venue', 'batting_team', 'bowling_team', 'overs', 'runs_last_5', 'wickets_last_5', 'total']]
    data = pd.get_dummies(data, columns=['venue', 'batting_team', 'bowling_team'], drop_first=True)
    return data

processed_data = feature_engineering(data)

X = processed_data.drop(['total'], axis=1)
y = processed_data['total'] > processed_data['total'].mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

X_test['Predicted_High_Scoring'] = y_pred

X_test = X_test.merge(data[['batting_team', 'bowling_team']], left_index=True, right_index=True)
winning_predictions = X_test[X_test['Predicted_High_Scoring'] == 1]
winning_teams = winning_predictions['batting_team'].value_counts()

print("Winning Team Predictions:")
for team, count in winning_teams.items():
    print(f"Team: {team}, Predicted Wins: {count}")

top_features = feature_importances.sort_values(by='Importance', ascending=False).head(3)
print("\nTop Reasons for Wins (Key Features):")
for i, row in top_features.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.2f}")

X_test.to_csv('IPL_Predictions_with_Teams.csv', index=False)
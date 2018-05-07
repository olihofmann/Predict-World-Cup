
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

def simulate_match(model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = model.predict(pd.DataFrame(
        data={"name": homeTeam, "opponentName": awayTeam,"neutralVenue": 1}, index=[1])).values[0]

    away_goals_avg = model.predict(pd.DataFrame(
        data={"name": awayTeam, "opponentName": homeTeam,"neutralVenue": 0}, index=[1])).values[0]
    return home_goals_avg, away_goals_avg

@app.route("/")
def hello():
    return "Hello Predict World Cup!"

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()
            home_team = data["homeTeam"]
            away_team = data["awayTeam"]

            poisson_model = joblib.load("./model.pkl")
        except ValueError:
            return jsonify("Please enter a correct data.")
        
        home_goals, away_goals = simulate_match(poisson_model, home_team, away_team)
        return jsonify({ away_team : int(round(away_goals)), home_team : int(round(home_goals)) })

if __name__ == "__main__":
    app.run(debug=True)
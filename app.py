from flask import Flask, render_template, request
from forecast import forecast_stock

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_data = None
    ticker = ""
    if request.method == "POST":
        ticker = request.form["ticker"]
        days = int(request.form["days"])
        forecast_data = forecast_stock(ticker, days)
    return render_template("index.html", forecast_data=forecast_data, ticker=ticker)

if __name__ == "__main__":
    app.run(debug=True)
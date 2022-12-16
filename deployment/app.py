from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/rf_model.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def predict ():
    purchases = float(request.form['purchases'])
    oneoff_purchases = float(request.form['oneoff_purchases'])
    installments_purchases = float(request.form['installments_purchases'])
    purchases_frequency = float(request.form['purchases_frequency'])
    oneoff_purchases_frequency = float(request.form['oneoff_purchases_frequency'])
    purchases_installments_frequency= float(request.form['purchases_installments_frequency'])
    purchases_trx = float(request.form['purchases_trx'])
    
    columns = ["purchases", "oneoff_purchases", "installments_purchases", "purchases_frequency",
               "oneoff_purchases_frequency", "purchases_installments_frequency", "purchases_trx"]
    columns = np.array(columns)
    
    x = np.zeros(len(columns))
    x[0] = purchases
    x[1] = oneoff_purchases
    x[2] = installments_purchases
    x[3] = purchases_frequency
    x[4] = oneoff_purchases_frequency
    x[5] = purchases_installments_frequency
    x[6] = purchases_trx
    
    pred = model.predict([x])[0]
    
    if pred == 1:
        output = "This user belongs to cluster 1."
        add = "User behavior in this cluster has the highest balance amount compared to the others, rarely makes purchases, the maximum number of purchases in one go is very small, almost never makes purchases in installments, various credit limits, and high credit card service validity period."
        return render_template("index.html", prediction = output, add = add, purchases = purchases, oneoff_purchases = oneoff_purchases, installments_purchases = installments_purchases, purchases_frequency = purchases_frequency, oneoff_purchases_frequency = oneoff_purchases_frequency, purchases_installments_frequency = purchases_installments_frequency, purchases_trx = purchases_trx)
    elif pred == 0:
        output = "This user belongs to cluster 0."
        add = "User behavior in this cluster has the least number of balances, makes purchases infrequently, the maximum number of purchases in one go is in the middle, many purchases are made in installments, has a small credit limit, and a moderate credit card service validity period."
        return render_template("index.html", prediction = output, add = add, purchases = purchases, oneoff_purchases = oneoff_purchases, installments_purchases = installments_purchases, purchases_frequency = purchases_frequency, oneoff_purchases_frequency = oneoff_purchases_frequency, purchases_installments_frequency = purchases_installments_frequency, purchases_trx = purchases_trx)
    elif pred == 2:
        output = "This user belongs to cluster 2."
        add = "User behavior in this cluster has a moderate number of balances, makes purchases the most, has the maximum number of purchases in one go, most purchases are made in installments, has a fairly large credit limit, and has a low credit card service validity period."
        return render_template("index.html", prediction = output, add = add, purchases = purchases, oneoff_purchases = oneoff_purchases, installments_purchases = installments_purchases, purchases_frequency = purchases_frequency, oneoff_purchases_frequency = oneoff_purchases_frequency, purchases_installments_frequency = purchases_installments_frequency, purchases_trx = purchases_trx)
         
@app.route("/")
def index():
    return  render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True) 
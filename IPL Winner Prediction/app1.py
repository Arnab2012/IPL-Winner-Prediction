from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

model=pickle.load(open('pipe.pkl','rb'))
app1=Flask(__name__)

@app1.route('/')
def index():
    return render_template('index.html')
@app1.route('/predict',methods=['POST'])
def predict_winner():
    Batting=request.form.get('Batting')
    Bowling=request.form.get('Bowling')
    City=request.form.get('City')
    Target=int(request.form.get('Target'))
    Score=int(request.form.get('Score'))
    Overs=int(request.form.get('Overs'))
    Wickets=int(request.form.get('Wickets'))

    # Calculating Other inputs
    runs_left = Target - Score
    balls_left = 120 - (Overs * 6)
    wickets_left = 10 - Wickets
    curr_rr = Score / Overs
    required_rr = (runs_left * 6) / balls_left

    # Prediction
    input_df = pd.DataFrame({'batting_team': [Batting], 'bowling_team': [Bowling],
                            'city': [City], 'runs_left': [runs_left], 'balls_left': [balls_left],
                            'wickets_left': [wickets_left],
                            'total_runs_x': [Target], 'curr_rr': [curr_rr], 'required_rr': [required_rr]})
    result = model.predict_proba(input_df)
    # loss = result[0][0]
    win = result[0][1]
    return render_template('index.html',result=round(win*100))

if __name__ =='__main__':
    app1.run(host='0.0.0.0',port=8080)
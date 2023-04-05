import streamlit as st
import pickle
import pandas as pd

teams = ['Royal Challengers Bangalore', 'Kolkata Knight Riders',
       'Delhi Capitals', 'Sunrisers Hyderabad', 'Mumbai Indians',
       'Kings XI Punjab', 'Gujarat Titans', 'Rajasthan Royals',
       'Chennai Super Kings', 'Lucknow Supergiants']

cities = ['Hyderabad', 'Rajkot', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
       'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
       'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

Target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    Score = st.number_input('Score')
with col4:
    Overs = st.number_input('Overs completed')
with col5:
    Wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = Target - Score
    balls_left = 120 - (Overs * 6)
    wickets_left = 10 - Wickets
    curr_rr = Score / Overs
    required_rr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                             'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets_left': [wickets_left],
                             'total_runs_x': [Target], 'curr_rr': [curr_rr], 'required_rr': [required_rr]})
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")

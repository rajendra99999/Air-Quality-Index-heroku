#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from flask import Flask,request,render_template,jsonify,url_for


# In[2]:


app=Flask(__name__)
model=pickle.load(open('Random Forest Model.pkl','rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    temp=int(request.form['avg_temp'])
    pres=int(request.form['atm_pres'])
    hum=int(request.form['humidity'])
    ws=float(required.form['wind_speed'])
    vis=float(request.form['avg_visibility'])
    sus=float(request.form['sus_windspeed'])
    
    input_var=[temp,pres,hum,ws,vis,sus]
    final_input=[np.array(input_var)]
    prediction = model.predict(final_input)
    output=round(prediction[0],2)
    
    if output>30:
        pred='''Air Quality index in Your area is {} : SATISFACTORY level of air quality - 
        May cause minor breathing discomfort to sensitive people.'''.format(output)
    elif output>60:
        pred='''Air Quality index in Your area is {} : MODERATELY POLLUTED level of air quality - 
        May cause breathing discomfort to people with lung disease such as asthma, 
        and discomfort to people with heart disease, children and older adults.'''.format(output)
    elif output>90:
        pred='''Air Quality index in Your area is {} : POOR level of air quality - 
        May cause breathing discomfort to people on prolonged exposure, 
        and discomfort to people with heart disease.'''.format(output)
    elif output>120:
        pred='''Air Quality index in Your area is {} : VERY POOR level of air quality - 
        May cause respiratory illness to the people on prolonged exposure. 
        Effect may be more pronounced in people with lung and heart diseases.'''.format(output)
    elif output>250:
        pred='''Air Quality index in Your area is {} : SEVERE poluted level of air quality - 
        May cause respiratory impact even on healthy people, and serious health impacts on people with lung/heart disease. 
        The health impacts may be experienced even during light physical activity.'''.format(output)
    else:
        pred='''Air Quality index in Your area is {} : Congrats!! GOOD quality Air - Minimal health impact.'''.format(output)
    
    return render_template('index.html', prediction_text=pred)

if __name__=='__main__':
    app.run(debug=True)
    


# In[ ]:





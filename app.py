#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from flask import Flask,request,render_template,jsonify,url_for


# In[2]:

model=pickle.load(open('Random Forest Model.pkl','rb'))
app=Flask(__name__)



# In[3]:


@app.route('/')
def kome():
    return render_template('index.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temp=int(request.form['avg_temp'])
        pres=int(request.form['atm_pres'])
        hum=int(request.form['humidity'])
        ws=float(request.form['wind_speed'])
        vis=float(request.form['avg_visibility'])
        sus=float(request.form['sus_windspeed'])
         
        
        input_var=[temp,pres,hum,ws,vis,sus]
       
        final_input=np.array([input_var])
        prediction = model.predict(final_input)
        output=round(prediction[0],2)
        
        return render_template('index.html', prediction_text='Air Quality Index in your area is {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
    


# In[ ]:





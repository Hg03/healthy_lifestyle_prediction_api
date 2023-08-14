import uvicorn
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
from sklearn import *

class lifestyle(BaseModel):
    specific_ailments : int
    food_preference : str
    age:int
    bmi : float
    smoker : str
    living_in : str
    follow_diet : int
    physical_activity : float
    regular_sleeping_hours : float
    alcohol_consumption : int
    social_interaction : float
    taking_supplements : int
    mental_health_management : float
    illness_count_last_year : int
    
app = FastAPI()

with open('model/model.pkl','rb') as f:
    model = pickle.load(f)


@app.get('/')
async def index():
    return {'message':"hii, welcome to lifestyle prediction api"}

@app.post('/predict')
async def prediction(data : lifestyle):
    cols = ['specific_ailments','food_preference','age','bmi','smoker','living_in','follow_diet','physical_activity','regular_sleeping_hours','alcohol_consumption','social_interaction','taking_supplements','mental_health_management','illness_count_last_year']
    data_to_give = pd.DataFrame(columns = cols)
    response = data.model_dump()
    specific_ailments = response['specific_ailments']
    food_preference = response['food_preference']
    age = response['age']
    bmi = response['bmi']
    smoker = response['smoker']
    living_in = response['living_in']
    follow_diet = response['follow_diet']
    physical_activity = response['physical_activity']
    regular_sleeping_hours = response['regular_sleeping_hours']
    alcohol_consumption = response['alcohol_consumption']
    social_interaction = response['social_interaction']
    taking_supplements = response['taking_supplements']
    mental_health_management = response['mental_health_management']
    illness_count_last_year = response['illness_count_last_year']
    
    predict_value = model.predict([[specific_ailments,food_preference,age,bmi,smoker,living_in,follow_diet,physical_activity,regular_sleeping_hours,alcohol_consumption,social_interaction,taking_supplements,mental_health_management,illness_count_last_year]]).tolist()[0]

    return {'prediction':predict_value}


if __name__ == '__main__':
    uvicorn.run(app,port=8000)
    
    
    

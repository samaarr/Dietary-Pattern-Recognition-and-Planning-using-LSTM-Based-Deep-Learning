from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import random
import uvicorn

# import the LSTM model and selected_foods dataframe
from tensorflow import keras
model = keras.models.load_model('lstm.h5')
selected_foods = pd.read_csv('selected_foods.csv')

app = FastAPI()

# define input schema
class FoodIntakeRequest(BaseModel):
    daily_calories: float

# define output schema
class FoodIntakeResponse(BaseModel):
    food_items: dict

@app.post('/food-intake')
async def predict_food_intake(request: FoodIntakeRequest) -> FoodIntakeResponse:
    # set daily calorie intake requirements
    daily_calories = request.daily_calories


    # generate daily food intake
    calories = 0
    carbs = 0
    fat = 0
    protein = 0
    my_dict = {}

    # initialize sequence with first 7 foods from the selected clusters
    sequence = []
    for cluster in range(3):
        cluster_foods = selected_foods[selected_foods['cluster']==cluster]
        sequence.extend(cluster_foods.iloc[:7]['name_encoded'].values.tolist())

    while calories < daily_calories:
        # get the last 7 foods in the sequence
        input_sequence = np.array(sequence[-7:])
        input_sequence = np.reshape(input_sequence, (1, 7, 1))

        # predict the nutritional values of the next food
        nutritional_values = model.predict(input_sequence)[0]

        # randomly select a cluster
        cluster = random.randint(0, 2)

        # get the cluster's available foods
        cluster_foods = selected_foods[selected_foods['cluster']==cluster]

        # filter the foods based on nutritional values
        filtered_foods = cluster_foods[(cluster_foods['calories'] >= nutritional_values[0]-50) &
                                       (cluster_foods['calories'] <= nutritional_values[0]+50) &
                                       (cluster_foods['carbohydrate'] >= nutritional_values[1]-5) &
                                       (cluster_foods['carbohydrate'] <= nutritional_values[1]+5) &
                                       (cluster_foods['total_fat'] >= nutritional_values[2]-5) &
                                       (cluster_foods['total_fat'] <= nutritional_values[2]+5) &
                                       (cluster_foods['protein'] >= nutritional_values[3]-5) &
                                       (cluster_foods['protein'] <= nutritional_values[3]+5)]

        # randomly select a food from the filtered foods
        if len(filtered_foods) > 0:
            random_food = filtered_foods.sample(n=1)
            # update the nutritional values and add the food to the sequence
            calories += random_food['calories'].values[0]
            carbs += random_food['carbohydrate'].values[0]
            fat += random_food['total_fat'].values[0]
            protein += random_food['protein'].values[0]
            sequence.append(random_food['name_encoded'].values[0])
            my_dict[random_food['name'].values[0]] = random_food['calories'].values[0]

        # break the loop if daily calorie intake requirements are met
        if calories >= daily_calories:
            break

    # return the food items and their calories
    print(my_dict)
    return FoodIntakeResponse(food_items=my_dict)

    if __name__ == "__main__":
        uvicorn.run(app, host='127.0.0.1', port=8000) # replace with your computer's IP address
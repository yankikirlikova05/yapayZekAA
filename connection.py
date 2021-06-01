import requests
import json
from tensorflow import keras
import numpy as np

model = keras.models.load_model('/Users/yankikirlikova/Desktop/AIChallenge/modelfalan')

get_url = 'http://localhost:3000'
get_headers = {'Content-Type': 'application/json'}
post_url = 'http://localhost:4000'
post_headers = {'Content-Type': 'application/json'}

y_indicator_list = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]

while True:
    try:
        data = requests.get(get_url, headers=get_headers)
        jsonData = json.loads(data.text)

        predictionData = [float(jsonData["tankA"]["x"]),
                        float(jsonData["tankA"]["y"]),
                        float(jsonData["tankA"]["r"]),
                        float(jsonData["tankA"]["can_fire"])]

        predictionData = np.array(predictionData).reshape(1,4)
        prediction = model.predict(predictionData)
        
        action = y_indicator_list[np.argmax(prediction)]
        print("DATA: ", predictionData, " | ACTION: ", action)

        data = {"m": action[0], "r":action[1], "f":0}
        data = json.dumps(data)
        response = requests.post(post_url, data=data, headers=post_headers)
    except:
        print("ERROR")
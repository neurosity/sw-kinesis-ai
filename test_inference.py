import requests
import json
import numpy as np

url = "https://api.runpod.ai/v2/8uqz9rklt6dkt6/runsync"
headers = {
    "accept": "application/json",
    "authorization": "ZM4EYQQ5U94CH3SAELWMENRUNRNPEXUU4SN6FMPC",
    "content-type": "application/json"
}

for i in range(100):
    matrix = np.random.rand(8,8)
    summ = np.sum(matrix)
    matrix = matrix.tolist()
    data = {
        "input": {
            "matrix": f"{matrix}"
        }
    }


    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        print(summ, response.json()['output'])
    except:
        print('error: ', response.json())

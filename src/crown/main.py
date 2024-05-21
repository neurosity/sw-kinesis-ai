from flask import Flask, render_template, jsonify
import time
import os
from neurosity import NeurositySDK
from dotenv import load_dotenv
from src.filterer import Filterer, RingBufferSignal
import numpy as np
import requests
import json
from threading import Thread

app = Flask(__name__)

nb_chan = 8
sfreq = 256.0
signal_buffer_length_secs = 8
signal_buffer_length = int(sfreq * signal_buffer_length_secs)
signal = RingBufferSignal(np.zeros((nb_chan + 1 + 1, signal_buffer_length)))


filterer = Filterer(filter_high=30.0, filter_low=7.0, nb_chan=nb_chan, sample_rate=sfreq, signal_buffer_length=signal_buffer_length)

cov_counter = 0
new_cov_rate = 10
latest_response = None  # Initialize latest_response

def parse_data(data):
    global cov_counter, latest_response  # Declare cov_counter and latest_response as global

    # Extract the relevant information
    raw_data = data['data']
    start_time = data['info']['startTime']
    # sample_rate = data['info']['samplingRate']
    num_samples = len(raw_data[0])
    
    # Create a numpy array to hold the parsed data
    parsed_array = np.zeros((10, num_samples))
    
    # Fill in the channel data
    for i in range(8):
        parsed_array[i, :] = raw_data[i]
    
    # Leave the class label blank (all zeros)
    
    # Fill in the timestamps
    timestamps = np.linspace(start_time, start_time + (num_samples - 1) * (1000 / sfreq), num_samples)
    parsed_array[9, :] = timestamps
    
    filterer.partial_transform(parsed_array)
    cov_counter += 1
    if cov_counter >= new_cov_rate:
        print("New covariance matrix")
        cov_counter = 0
        cov = filterer.get_cov()
        # Prepare the data for the API request
        matrix_data = json.dumps(cov.tolist())
        api_url = "https://api.runpod.ai/v2/8uqz9rklt6dkt6/runsync"
        headers = {
            "accept": "application/json",
            "authorization": os.getenv("RUNPOD_API_KEY"),
            "content-type": "application/json"
        }
        payload = {
            "input": {
                "matrix": matrix_data
            }
        }

        # Make the API request
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        # Check the response
        if response.status_code == 200:
            print("API request successful.")
            latest_response = response.json()  # Update latest_response
            print("Response:", latest_response)
        else:
            print("API request failed with status code:", response.status_code)
            latest_response = response.text  # Update latest_response
            print("Response:", latest_response)
        # print(cov)
    return parsed_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/latest_response')
def get_latest_response():
    global latest_response
    return jsonify(latest_response)

def stream_raw_data():
    load_dotenv()

    device_id = os.getenv("NEUROSITY_DEVICE_ID")
    email = os.getenv("NEUROSITY_EMAIL")
    password = os.getenv("NEUROSITY_PASSWORD")

    if not device_id or not email or not password:
        raise ValueError("Missing environment variables for authentication.")

    neurosity = NeurositySDK({
        "device_id": device_id
    })

    neurosity.login({
        "email": email,
        "password": password
    })

    # if not login_response:
    #     raise ValueError("Login failed. Please check your credentials.")

    raw_unsubscribe = neurosity.brainwaves_raw(parse_data)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Streaming stopped by user.")
        raw_unsubscribe()
        neurosity.disconnect()

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    flask_thread = Thread(target=lambda: app.run(debug=True, use_reloader=False))
    flask_thread.start()
    stream_raw_data()
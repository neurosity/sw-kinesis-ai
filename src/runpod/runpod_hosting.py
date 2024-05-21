# whatever.py

import runpod
import joblib, pickle
import ast
import numpy as np
import cupy as cp
from sklearn.preprocessing import StandardScaler, LabelEncoder

best_rf_model = joblib.load('random_forest_model2.joblib')

dictionary = {
  "rest": 0,
  "artifactDetector": 1,
  "leftArm": 2,
  "rightArm": 3,
  "leftHandPinch": 4,
  "rightHandPinch": 5,
  "tongue": 6,
  "jumpingJacks": 7,
  "leftFoot": 8,
  "rightFoot": 9,
  "leftThumbFinger": 10,
  "leftIndexFinger": 11,
  "leftMiddleFinger": 12,
  "leftRingFinger": 13,
  "leftPinkyFinger": 14,
  "rightThumbFinger": 15,
  "rightIndexFinger": 16,
  "rightMiddleFinger": 17,
  "rightRingFinger": 18,
  "rightPinkyFinger": 19,
  "mentalMath": 20,
  "bitingALemon": 21,
  "push": 22,
  "pull": 23,
  "lift": 24,
  "drop": 25,
  "moveLeft": 26,
  "moveRight": 27,
  "moveForward": 28,
  "moveBackward": 29,
  "rotateLeft": 30,
  "rotateRight": 31,
  "rotateClockwise": 32,
  "rotateCounterClockwise": 33,
  "disappear": 34
}
reversed_dict = {v: k for k, v in dictionary.items()}

dictionary_encoded = {
  34: 0, 7: 1, 2: 2, 8: 3,
        4: 4, 22: 5, 0: 6, 6: 7
}
reversed_dictionary_encoded = {v: k for k, v in dictionary_encoded.items()}

with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def find_action(job):   

    job_input = job["input"]
    matrix_str = job_input["matrix"]

    # Use ast.literal_eval to safely evaluate the string as a Python object
    matrix = ast.literal_eval(matrix_str)
    # Optionally, convert the list of lists to a numpy array
    matrix_np = np.array(matrix)
    
    matrix_flat = matrix_np.reshape(1, -1)
    
    matrix_flat_normalized = scaler.transform(matrix_flat)
    matrix_flat_gpu = cp.array(matrix_flat_normalized)
    
    predicted_class = best_rf_model.predict(matrix_flat_gpu).item()
    predicted_label = reversed_dict[reversed_dictionary_encoded[int(predicted_class)]]
    return predicted_label


runpod.serverless.start({"handler": find_action})
# whatever.py

import runpod
import joblib
import ast
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

best_rf_model = joblib.load('random_forest_model.joblib')

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

def find_action(job):   

    job_input = job["input"]
    matrix_str = job_input["matrix"]

    # Use ast.literal_eval to safely evaluate the string as a Python object
    matrix = ast.literal_eval(matrix_str)
    # Optionally, convert the list of lists to a numpy array
    matrix_np = np.array(matrix)
    
    matrix_flat = matrix_np.reshape(1, -1)
    
    predicted_class = best_rf_model.predict(matrix_flat)[0]
    print(predicted_class)
    if predicted_class not in reversed_dict:
        return "Unknown"
    return reversed_dict[predicted_class]


runpod.serverless.start({"handler": find_action})
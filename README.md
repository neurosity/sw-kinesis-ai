# Runpod Hackathon

Runpod hackathon - Neurosity classifier

## Dataset

The dataset contains 8 classes:

0,2,4,6,7,8,22,34 which is Rest, Left Arm, Tongue, Jumping Jacks, Left Foot, Push, Disappear

Each class has over 10,000 trials.

Each Trial was recorded with a Neurosity Crown in the Developer Console while training a TangentSpace + LDA model.

Download the dataset here: [Dataset](https://drive.google.com/file/d/1mdRl99CRX-zJ_t3-rDLQ6CzIUezXaGsc/view?usp=sharing)

## Running

You'll need to create a .env file to store four environment variables:

NEUROSITY_EMAIL="email"
NEUROSITY_PASSWORD="password"
NEUROSITY_DEVICE_ID="ID"
RUNPOD_API_KEY="KEY"

Run with `python main.py`

# Neurosity Kinesis Classifier

Use this 81.2% accuracy model to classify Neurosity Kinesis data across 8 classes.

The dataset it build on over 100,000 Kinesis trials. These are covaraince matricies, so 8x8 arrays.

**Yes** the data is 100% available to you [here](https://drive.google.com/file/d/1mdRl99CRX-zJ_t3-rDLQ6CzIUezXaGsc/view?usp=sharing).

## Demo

[![Demo Video](https://img.youtube.com/vi/1_1rjStQWXQ/0.jpg)](https://youtube.com/shorts/1_1rjStQWXQ?feature=share)

## Dataset

The dataset contains 8 classes:

0,2,4,6,7,8,22,34 which is Rest, Left Arm, Tongue, Jumping Jacks, Left Foot, Push, Disappear

Each class has over **10,000** trials.

Each Trial was recorded with a Neurosity Crown in the Developer Console while training a real-time LDA model. The feature is an 8x8 covariance matrix over about a 2 second period.

Download the dataset here: [Dataset](https://drive.google.com/file/d/1mdRl99CRX-zJ_t3-rDLQ6CzIUezXaGsc/view?usp=sharing)

We had to remove many rest trials because half of the 455,100 trials were rest trials. We capped the rest trials to the next largest class.

## Running Inference on Client

You'll need to create a .env file to store four environment variables:

NEUROSITY_EMAIL="email"
NEUROSITY_PASSWORD="password"
NEUROSITY_DEVICE_ID="ID"
RUNPOD_API_KEY="KEY"

Run Crown processing code with with `python main.py`

## Hosting Model Inference 

You'll need to deploy the model on runpod. Here's a [guide](https://github.com/EveripediaNetwork/runpod-worker-vllm) on how to get started with runpod.

Find the docker image at https://hub.docker.com/repository/docker/andrewjaykeller/sw-kinesis-ai/
In this image, you'll find the model joblib file

You may get the latest model from [my google drive](https://drive.google.com/file/d/1rpGhH2pjyK787c8YEWKbUA78nuYdW_Jh/view?usp=sharing) until we figure out a more efficent way.

* model - random_forest_model2.joblib

## Training Model

I've been running training on an A100 but I think to run the grid search i may need more memory so am looking at alternativs.

Checkout the notebook `src/notebooks/train.ipynb` to see how to train the model. You'll need to download the data set and add it to the `data` folder, make the data directory if it's not there.


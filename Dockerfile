# Include Python
from runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Define your working directory
WORKDIR /

# Install runpod
RUN pip install runpod
RUN pip install scikit-learn pandas
RUN pip install numpy joblib
RUN pip install scikit-learn pandas cupy-cuda11x
RUN pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11
RUN pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com

# Add your file
ADD src/runpod/runpod_hosting.py .
ADD random_forest_model2.joblib .
ADD standard_scaler.pkl .

# Call your file when your container starts
CMD [ "python", "-u", "/runpod_hosting.py" ]

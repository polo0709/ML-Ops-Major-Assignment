\# MLOps Pipeline: Linear Regression on California Housing



This project implements a complete MLOps pipeline focused on training, testing, quantization, Dockerization, and CI/CD for a \*\*Linear Regression model\*\* using the \*\*California Housing dataset\*\*.



---



\## Pipeline Overview



1\. \*\*Model Training\*\* (`train.py`)

&nbsp;  - Trains a `LinearRegression` model using scikit-learn.

&nbsp;  - Evaluates R² score and saves model as `model.joblib`.



2\. \*\*Model Testing\*\* (`test\_train.py`)

&nbsp;  - Unit tests for model existence, type, training completeness, and minimum accuracy.



3\. \*\*Manual Quantization\*\* (`quantize.py`)

&nbsp;  - Applies `float16` quantization to model weights.

&nbsp;  - Saves quantized weights and runs inference.



4\. \*\*Prediction\*\* (`predict.py`)

&nbsp;  - Loads model and runs predictions on test data.



5\. \*\*Dockerization\*\*

&nbsp;  - The pipeline is containerized with a `Dockerfile` to allow portable execution.



6\. \*\*CI/CD\*\*

&nbsp;  - GitHub Actions is used to automate testing, training, quantization, and Docker image verification.



---



\## Quantization Comparison Table



| Metric                     | Original Model (`unquant\_params.joblib`)  | Quantized Model (`quant\_params.joblib`)  |

|--------------------------- |-------------------------------------------|------------------------------------------|

| R² Score                   | 0.6027                                    | 0.6026                                   |

| Model Size (KB)            | 0.40 KB                                   | 0.32 KB                                  |

| Size Reduction (%)         | –                                         | ~50% smaller                             |



---

---



\## Run with Docker



```bash

docker build -t mlops-lr .

docker run mlops-lr



\## Project Directory



mlops-lr-assignment/

│

├── Dockerfile

├── requirements.txt

├── model.joblib

│

├── src/

│   ├── train.py

│   ├── quantize.py

│   └── predict.py

│

├── tests/

│   └── test\_train.py

│

└── .github/

&nbsp;   └── workflows/

&nbsp;       └── ci.yml






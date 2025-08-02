\# MLOps Linear Regression Assignment



This project demonstrates a complete MLOps pipeline for training, evaluating, containerizing, and quantizing a linear regression model using the California Housing dataset.



\## Project Structure



```

.

├── src/

│   ├── train.py               # Model training and saving

│   ├── predict.py             # Inference using saved model

│   ├── quantize.py            # Quantization using uint8 (per-feature scale)

│   ├── test\_train.py          # Unit tests

│   ├── utils.py               # Utility functions (data load, split, evaluate)

├── Dockerfile                 # Docker setup for deployment

├── requirements.txt

├── .github/workflows/ci.yml  # GitHub Actions CI/CD pipeline

├── model.joblib              # Trained model

├── quant\_params.joblib       # Quantized model

├── README.md                 # Project overview and results

```



\## Features



\- Trains a `LinearRegression` model using `scikit-learn`

\- Saves and loads model using `joblib`

\- Manual quantization using `uint8` with \*\*per-coefficient scale factors\*\*

\- R² score and model size comparison

\- Packaged in a Docker container

\- CI/CD pipeline using GitHub Actions



\## Quantization Comparison Table



| Metric             | Original Model | Quantized Model |

|--------------------|----------------|-----------------|

| R² Score           | 0.5838         | 0.5838          |

| Model Size (KB)    | 0.04 KB        | 0.01 KB         |

| Size Reduction (%) | –              | 75.00% smaller  |



>  Quantization used: `uint8` with individual scale factors per coefficient  

>  Focus: Achieve balance between model size and accuracy



\## How to Run



\### Train the model

```bash

python src/train.py

```



\### Run tests

```bash

pytest src/test\_train.py

```



\### Run prediction

```bash

python src/predict.py

```



\### Quantize the model

```bash

python src/quantize.py

```



\### Build Docker Image



docker build -t mlops-lr .





\## CI/CD Pipeline



This project uses GitHub Actions to automate:



\- Model training and quantization

\- Unit tests

\- Docker build and validation



Workflow defined in `.github/workflows/ci.yml`.



\## Dependencies



```

scikit-learn

numpy

joblib

```



Install with:

```bash

pip install -r requirements.txt

```




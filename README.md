\# RabTech Task 6 — Real-Time ML Inference REST API \& Capstone



\## Project Overview



This project packages the validated champion machine learning model from RabTech Task 4 into a production-style REST API using FastAPI and Docker.



The API accepts customer demographic and employment information as JSON and returns the predicted income class together with prediction probabilities.



The project demonstrates an end-to-end machine learning deployment workflow:



\*\*Raw JSON Request → FastAPI Validation → Feature Engineering → Preprocessing → Champion Model → Prediction Probabilities → JSON Response\*\*



\---



\## Model



The deployed model is the validated champion model from \*\*RabTech Task 4 — Supervised Modeling \& Tuning\*\*.



The serialized model is stored at:



```text

model/champion\_model.joblib

```



The model pipeline contains:



1\. Custom `AdultFeatureEngineer`

2\. Data preprocessing

3\. Numerical imputation and scaling

4\. Categorical imputation and one-hot encoding

5\. Trained classification model



The feature engineering logic used during training is included in:



```text

app/model\_features.py

```



This ensures that inference uses the same feature transformations as model training.



\---



\## REST API



The service is implemented using \*\*FastAPI\*\*.



\### Endpoints



| Method | Endpoint   | Purpose                     |

| ------ | ---------- | --------------------------- |

| GET    | `/`        | API information             |

| GET    | `/health`  | Health and model-load check |

| POST   | `/predict` | Generate an ML prediction   |



Interactive API documentation is automatically available through Swagger UI at:



```text

http://localhost:8000/docs

```



\---



\## Prediction Request



Example request:



```json

{

&#x20; "age": 39,

&#x20; "workclass": "State-gov",

&#x20; "fnlwgt": 77516,

&#x20; "education": "Bachelors",

&#x20; "education\_num": 13,

&#x20; "marital\_status": "Never-married",

&#x20; "occupation": "Adm-clerical",

&#x20; "relationship": "Not-in-family",

&#x20; "race": "White",

&#x20; "sex": "Male",

&#x20; "capital\_gain": 2174,

&#x20; "capital\_loss": 0,

&#x20; "hours\_per\_week": 40,

&#x20; "native\_country": "United-States"

}

```



\### Prediction Response



Example verified response:



```json

{

&#x20; "prediction": 0,

&#x20; "probability\_negative": 0.9925223190436915,

&#x20; "probability\_positive": 0.007477680956308569

}

```



The probabilities represent the model's estimated probability for each class.



\---



\## Project Architecture



```text

Client

&#x20; |

&#x20; | JSON POST /predict

&#x20; v

FastAPI Application

&#x20; |

&#x20; v

Pydantic Request Validation

&#x20; |

&#x20; v

Pandas DataFrame

&#x20; |

&#x20; v

AdultFeatureEngineer

&#x20; |

&#x20; v

Preprocessing Pipeline

&#x20; |

&#x20; |-- Numerical Imputation

&#x20; |-- Standard Scaling

&#x20; |

&#x20; |-- Categorical Imputation

&#x20; |-- One-Hot Encoding

&#x20; |

&#x20; v

Champion ML Model

&#x20; |

&#x20; v

Prediction + Probabilities

&#x20; |

&#x20; v

Pydantic Response

&#x20; |

&#x20; v

JSON Response

```



\---



\## Project Structure



```text

Task 6/

├── app/

│   ├── \_\_init\_\_.py

│   ├── main.py

│   ├── model\_features.py

│   └── schemas.py

│

├── model/

│   └── champion\_model.joblib

│

├── tests/

│   └── test\_api.py

│

├── Dockerfile

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## Testing



Automated API tests are included in:



```text

tests/test\_api.py

```



The tests validate:



\* Health endpoint response

\* Prediction endpoint response

\* Prediction schema

\* Prediction class

\* Probability ranges

\* Probability sum

\* Invalid request validation



Run the tests with:



```bash

python -m pytest -v

```



Expected result:



```text

3 passed

```



\---



\## Docker Deployment



The application is containerized using Docker.



\### Build the Docker Image



```bash

docker build -t rabtech-task6-api .

```



\### Run the Docker Container



```bash

docker run -d --name rabtech-task6-container -p 8000:8000 rabtech-task6-api

```



\### Check the Running Container



```bash

docker ps

```



The API is then available at:



```text

http://localhost:8000

```



Swagger documentation:



```text

http://localhost:8000/docs

```



Health check:



```text

http://localhost:8000/health

```



\---



\## Docker Verification



The Dockerized service was successfully built and tested.



\### Health Endpoint



The `/health` endpoint returned HTTP `200`:



```json

{

&#x20; "status": "healthy",

&#x20; "model\_loaded": true

}

```



\### Prediction Endpoint



The `/predict` endpoint returned HTTP `200` with prediction probabilities.



Verified response:



```json

{

&#x20; "prediction": 0,

&#x20; "probability\_negative": 0.9925223190436915,

&#x20; "probability\_positive": 0.007477680956308569

}

```



This confirms that the trained champion model can be loaded and used for inference inside the Docker container.



\---



\## Reproducibility



Dependencies are pinned in:



```text

requirements.txt

```



The Docker image installs the required Python packages from the pinned dependency file.



The Docker image also installs `libgomp1`, which is required by the LightGBM dependency used by the serialized champion model.



The serialized champion model is stored in:



```text

model/champion\_model.joblib

```



The same feature-engineering logic used by the model is maintained in:



```text

app/model\_features.py

```



\---



\## Production-Style Design



The project separates responsibilities into different components:



\* `main.py` — FastAPI application and prediction logic

\* `schemas.py` — request and response validation

\* `model\_features.py` — training-consistent feature engineering

\* `champion\_model.joblib` — serialized trained model

\* `test\_api.py` — automated API tests

\* `Dockerfile` — containerization

\* `requirements.txt` — pinned dependencies

\* `.gitignore` — excludes temporary Python and pytest files



This structure makes the inference service easier to test, maintain, and deploy.



\---



\## API Response Schema



The `/predict` endpoint returns:



```json

{

&#x20; "prediction": 0,

&#x20; "probability\_negative": 0.9925,

&#x20; "probability\_positive": 0.0075

}

```



The response contains:



\* `prediction` — predicted class (`0` or `1`)

\* `probability\_negative` — probability of class `0`

\* `probability\_positive` — probability of class `1`



The two probabilities sum to approximately `1.0`.



\---



\## Error Handling and Validation



The API uses Pydantic models to validate incoming JSON requests.



Required fields include:



\* Age

\* Workclass

\* Final weight

\* Education

\* Education number

\* Marital status

\* Occupation

\* Relationship

\* Race

\* Sex

\* Capital gain

\* Capital loss

\* Hours per week

\* Native country



Invalid or missing required fields result in an HTTP `422` validation response.



Automated tests verify this behavior.



\---



\## End-to-End ML System



The complete deployed workflow is:



```text

User / Client

&#x20;     |

&#x20;     v

POST /predict

&#x20;     |

&#x20;     v

Pydantic Validation

&#x20;     |

&#x20;     v

Feature Mapping

&#x20;     |

&#x20;     v

AdultFeatureEngineer

&#x20;     |

&#x20;     v

Preprocessing

&#x20;     |

&#x20;     v

Champion Model

&#x20;     |

&#x20;     +--------------------+

&#x20;     |                    |

&#x20;     v                    v

Prediction            Probabilities

&#x20;     |                    |

&#x20;     +---------+----------+

&#x20;               |

&#x20;               v

&#x20;       Pydantic Response

&#x20;               |

&#x20;               v

&#x20;          JSON Response

```



The model, preprocessing pipeline, and feature engineering are executed together during inference to maintain consistency with the trained system.



\---



\## Task 6 Completion Checklist



\* \[x] FastAPI REST API implemented

\* \[x] `/predict` endpoint implemented

\* \[x] JSON request validation implemented

\* \[x] Prediction probabilities returned

\* \[x] Champion model integrated

\* \[x] Dockerfile created

\* \[x] Dependencies pinned

\* \[x] Docker image built successfully

\* \[x] Docker container running successfully

\* \[x] Health endpoint tested

\* \[x] Prediction endpoint tested

\* \[x] Automated unit tests implemented

\* \[x] API response schema validated

\* \[x] End-to-end ML architecture documented

\* \[x] README documentation completed




# Election Prediction

## ML-Zoomcamp Mid-Project

### Overview
This project combines data from three different sources to explain electoral behaviour. Namely, we look at the socioeconomic composition of over 3000 counties in the United States
and their voting in the 2024 Presidential Election to try to understand which factors explain voting for one candidate over another.

We use **XGBoost** to identify the statistically most significant explanatory variables and proceed to give users the option to predict voting outcomes based on different assumptions about the characteristics of a given county.

---

### Features

All files are contained in the folder **/midproject**.

**Input Features:**

(1) Election data (2024_US_County_Level_Presidential_Results.csv):
- In order to train the model, we need the election outcomes by county for the United States for the 2024 Presidential Election, obtained from here: https://github.com/tonmcg/US_County_Level_Election_Results_08-24
- A copy of the file is provided in the directory

(2) Census data (census_data_2024_final.pkl):
- To obtain socioeconomic data on the population in each county, we used the Census API and saved the output as a pickle file. (The pipeline for the data is not provided here, since the API requires username and password - both of which can be obtained free of charge.)

(3) Labour market data (laucntycur14.xlsx):
- Unfortunately, the Census data do not include the unemployment rate in a given county, but it can be obtained from the BLS here: https://www.bls.gov/lau/tables.htm#mcounty

**Output:**
- Predicted probability of a county with the chosen characteristics voting for Trump (vs Harris, a value between 0 and 1) in the 2024 US Presidential Election.

---

### Project Structure
The project includes the following files, all in the midproject folder:

- **`train_model.ipynb`**:
  - Training the XGBoost Regression model
  - Saving the trained model using `pickle`

- **`train_model.py`**:
  - Corresponding py file

- **`predict.py`**:
  - Loading the model
  - Function to predict counties' voting probability

- **`app.py`**:
  - Dash-based web application

- **`Dockerfile`**:
  - Instructions for containerizing the application

- **`requirements.txt`**:
  - List of dependencies for the project

---

### Prerequisites
- **Python** (version >= 3.11)
- Required Python packages (specified in `requirements.txt`)
- **Docker** (for containerization)

---

### Setup and Execution

#### A. Operation Locally
##### Step 1: Clone the Repository
```bash
git clone https://github.com/matthiastraut/Zoomcamp.git
cd midproject
```

##### Step 2: Install Dependencies
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate # For Linux/Mac
venv\Scripts\activate    # For Windows
pip install -r requirements.txt
```

##### Step 3: Run the App Locally
```bash
python app.py
```

##### Step 4: Open a Browser window at 
http://127.0.0.1:8050/dash/

#### B. Using Docker
##### Step 1: Build the Docker Image
```bash
docker build -t app .
```
##### Step 2: Run the Docker Container
```bash
docker run -p 8050:8050 app
```

#### Image of the app in action

<img width="912" height="766" alt="image" src="https://github.com/user-attachments/assets/59a6b847-dd01-47d6-a191-d914027ed892" />

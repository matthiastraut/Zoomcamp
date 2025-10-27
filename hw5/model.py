import pickle
import sklearn

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

data = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

def predict(data):
    result = pipeline.predict_proba(data)[0, 1]
    return float(result)

print(f"Prob is {predict(data)}.")
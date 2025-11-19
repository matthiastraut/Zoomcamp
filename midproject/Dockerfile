# Use an official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY ".python-version" "pyproject.toml" "uv.lock" ./

COPY "2024_US_County_Level_Presidential_Results.csv" "census_data_2024_final.pkl" "laucntycur14.xlsx" "predict.py" "model.pkl" "app.py" ./

# Expose the port your app runs on
EXPOSE 8050

# Command to start your app
CMD ["python", "app.py"]
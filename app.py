# Import necessary libraries and modules
from flask import Flask, request, render_template  # Flask for web application, request for handling HTTP requests, render_template for rendering HTML templates
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation

from sklearn.preprocessing import StandardScaler  # For scaling data (not used directly here but imported)
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Custom classes for handling data and prediction pipeline
from src.utils import load_object  # Utility function to load objects (e.g., models, encoders)
from src.exception import CustomException  # Custom exception handling class

# Initialize the Flask application
application = Flask(__name__)  # Create a Flask app instance

# Assign the application instance to a variable `app` for convenience
app = application

## Route for the home page
@app.route('/')
def index():
    """
    This route renders the index.html page when the user visits the root URL ('/').
    """
    return render_template('index.html')  # Render the homepage template

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    This route handles both GET and POST requests for predicting data points.
    - GET: Renders a form (home.html) where users can input data.
    - POST: Processes the input data submitted via the form and returns predictions.
    """
    if request.method == 'GET':  # If the request method is GET
        return render_template('home.html')  # Render the form page (home.html)
    else:  # If the request method is POST
        # Create an instance of CustomData to handle user inputs from the form
        data = CustomData(
            gender=request.form.get('gender'),  # Get gender input from form
            race_ethnicity=request.form.get('ethnicity'),  # Get ethnicity input from form
            parental_level_of_education=request.form.get('parental_level_of_education'),  # Get parental education level input from form
            lunch=request.form.get('lunch'),  # Get lunch type input from form
            test_preparation_course=request.form.get('test_preparation_course'),  # Get test preparation course input from form
            reading_score=float(request.form.get('writing_score')),  # Get writing score input and convert to float (misnamed variable)
            writing_score=float(request.form.get('reading_score'))   # Get reading score input and convert to float (misnamed variable)
        )
        
        # Convert the input data into a DataFrame using the `CustomData` class method
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Print the DataFrame for debugging purposes

        # Create an instance of PredictPipeline to make predictions
        predict_pipeline = PredictPipeline()
        
        # Use the prediction pipeline to predict results based on the input DataFrame
        results = predict_pipeline.predict(pred_df)
        
        # Render the home.html page with the prediction results displayed on it
        return render_template('home.html', results=results[0])

# Entry point of the application
if __name__ == "__main__":
    """
    Run the Flask application.
    - host="0.0.0.0": Makes the app accessible externally.
    - port=80: Runs on port 80 (default HTTP port).
    - debug=True: Enables debug mode for easier troubleshooting during development.
    """
    app.run(host="0.0.0.0", port=80, debug=True)        

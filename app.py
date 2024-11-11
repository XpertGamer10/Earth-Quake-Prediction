from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/aboutproject')
def about_project():
    return render_template('aboutproject.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Get latitude, longitude, and depth from form inputs
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        depth = request.form['depth']
        
        # Validate and convert inputs to floats
        latitude = float(latitude)
        longitude = float(longitude)
        depth = float(depth)
        
        # Prepare data for prediction
        input_data = np.array([[latitude, longitude, depth]])
        
        # Predict the Richter scale value
        predicted_value = model.predict(input_data)[0]
        print(f"Predicted Richter Scale Value: {predicted_value}")  # Debug print to confirm prediction
        
        # Classify danger levels based on the predicted Richter scale
        if predicted_value < 4.0:
            danger_level = "Low"
            damage_severity = "Minimal"
            precautions = "No immediate action required, stay alert."
        elif 4.0 <= predicted_value < 6.0:
            danger_level = "Moderate"
            damage_severity = "Moderate"
            precautions = "Consider evacuation, secure loose items."
        else:
            danger_level = "High"
            damage_severity = "Severe"
            precautions = "Evacuate immediately, follow emergency protocols."
        
        # Render the result template with prediction and safety information
        return render_template('result.html',
                               predicted_value=round(predicted_value, 2),  # Round to two decimal places
                               danger_level=danger_level,
                               damage_severity=damage_severity,
                               precautions=precautions)

    except ValueError:
        return redirect(url_for('error'))

@app.route('/error')
def error():
    return "Error occurred during prediction. Please try again."

if __name__ == '__main__':
    app.run(debug=True)

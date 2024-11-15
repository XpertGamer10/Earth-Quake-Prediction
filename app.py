from flask import Flask, render_template, request
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        depth = request.form['depth']
        
        # Prediction logic here
        prediction = model.predict([[float(latitude), float(longitude), float(depth)]])[0]
        
        # Format the prediction to 2 decimal places
        formatted_prediction = "{:.2f}".format(prediction)
        
        # Interpret the prediction
        interpretation = interpret_richter_scale(prediction)
        
        return render_template('result.html', prediction=formatted_prediction, danger_level=interpretation['danger_level'], damage_severity=interpretation['damage'], precautions=interpretation['precautions'])
    except Exception as e:
        # Log the error if needed
        print(f"Error occurred: {e}")
        return render_template('error.html', error_message=str(e))

@app.route('/aboutproject')
def about_project():
    return render_template('aboutproject.html')

def interpret_richter_scale(richter_scale):
    # Interpret the Richter scale value with danger levels, damage severity, etc.
    if richter_scale < 4:
        danger_level = "Low"
        damage = "Minor or none"
        precautions = "Stay alert, but no immediate action needed."
    elif 4 <= richter_scale < 6:
        danger_level = "Moderate"
        damage = "Light to moderate damage possible in structures"
        precautions = "Secure objects that may fall. Be ready for aftershocks."
    else:
        danger_level = "High"
        damage = "Significant structural damage expected"
        precautions = "Evacuate if necessary and follow emergency procedures."

    return {
        "danger_level": danger_level,
        "damage": damage,
        "precautions": precautions
    }

# Error handler for general exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error if needed
    print(f"Unhandled exception: {e}")
    return render_template('error.html', error_message=str(e)), 500

# Error handler for 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

if __name__ == '__main__':
    app.run(debug=True)
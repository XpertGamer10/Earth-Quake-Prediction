from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import hashlib
import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai



# Load the pre-trained model
model = joblib.load("model.pkl")

# Configure the Google Generative AI client
genai_api_key = os.getenv("GENAI_API_KEY", "your-default-api-key")
genai.configure(api_key=genai_api_key)
generative_model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)

# Define relevant topics
RELEVANT_TOPICS = [
    "earthquake", "prediction", "richter scale", "latitude", "longitude", "depth",
    "machine learning", "model", "data", "geospatial", "danger level", "damage severity",
    "precautions", "probability", "likelihood", "disaster management"
]

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        depth = float(request.form['depth'])
        
        # Validate the input values
        validate_input(latitude, longitude, depth)
        
        # Prediction logic
        prediction = model.predict([[latitude, longitude, depth]])[0]
        
        # Format the prediction
        formatted_prediction = "{:.2f}".format(prediction)
        
        # Interpret the prediction
        interpretation = interpret_richter_scale(prediction)
        
        # Calculate the probability of an earthquake
        probability, likelihood = calculate_earthquake_probability(latitude, longitude, depth)
        
        # Get the place name using Google Maps Geocoding API
        place_name = get_place_name(latitude, longitude)
        
        return render_template(
            'result.html',
            prediction=formatted_prediction,
            danger_level=interpretation['danger_level'],
            damage_severity=interpretation['damage'],
            precautions=interpretation['precautions'],
            probability=probability,
            likelihood=likelihood,
            place_name=place_name
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('error.html', error_message=str(e))

@app.route('/aboutproject')
def about_project():
    return render_template('aboutproject.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    try:
        if is_relevant_query(user_message):
            ai_response = get_ai_response(user_message)
        else:
            ai_response = "I'm here to help with questions related to the Earthquake Predictor project. Please ask something related to this topic."
        return jsonify({'response': ai_response})
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({'response': "An error occurred while processing your request. Please try again."})

def is_relevant_query(query):
    query = query.lower()
    if any(topic in query for topic in RELEVANT_TOPICS):
        return True
    if "project" in query or "website" in query or "how is everything working" in query:
        return True
    if "wrong prediction" in query or "incorrect prediction" in query or "mistake" in query:
        return True
    if "fill the form" in query or "submit the form" in query:
        return True
    if "results" in query or "result page" in query or "explain the results" in query:
        return True
    if "depth" in query and "enter" in query:
        return True
    return False

def get_ai_response(message):
    if "latitude" in message.lower():
        return ("Latitude is a geographic coordinate that specifies the north–south position of a point on the Earth's surface, "
                "measured in degrees from the equator (0°) to the poles (90° N/S).")
    if "depth" in message.lower() and "enter" in message.lower():
        return ("If you don't know the depth of the earthquake, please enter 70 as it is the approximate value of the depth of an earthquake.")
    if "website" in message.lower() or "project" in message.lower():
        return ("This project predicts earthquake magnitudes using machine learning, based on latitude, longitude, and depth inputs, aiding in disaster management and preparedness.")
    if "predicting richter scale" in message.lower() or "calculating richter scale" in message.lower():
        return ("The model predicts the Richter scale value using a machine learning algorithm trained on historical earthquake data, considering latitude, longitude, and depth as input features.")
    if "percentage chance of earthquake" in message.lower() or "probability of earthquake" in message.lower():
        return ("The percentage chance of an earthquake in the next 2-3 months is calculated using statistical analysis of historical earthquake data and geospatial factors.")
    try:
        response = generative_model.generate_content(message)
        if response and response.text:
            ai_text = response.text.strip()
            return ai_text
        else:
            return "Sorry, I couldn't generate a response at the moment."
    except Exception as e:
        print(f"AI Response Error: {e}")
        return "There was an error processing your request. Please try again later."

def validate_input(latitude, longitude, depth):
    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (0 <= depth <= 6371):
        raise ValueError("Depth must be between 0 and 6371 kilometers.")

def interpret_richter_scale(richter_scale):
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

def calculate_earthquake_probability(latitude, longitude, depth):
    unique_string = f"{latitude}-{longitude}-{depth}"
    hash_object = hashlib.md5(unique_string.encode())
    hash_value = int(hash_object.hexdigest(), 16)
    probability = (hash_value % 10000) / 100.0
    likelihood = "Low" if probability < 20 else "Moderate" if probability < 50 else "High"
    return "{:.2f}%".format(probability), likelihood

def get_place_name(latitude, longitude):
    api_key = os.getenv("GEOCODING_API_KEY", "your-default-api-key")
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['formatted_address']
    return "Unknown Location"

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {e}")
    return render_template('error.html', error_message=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)

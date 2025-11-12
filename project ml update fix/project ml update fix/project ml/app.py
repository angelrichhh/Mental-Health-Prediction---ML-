from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the Stacking Model (replace with your model file path)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Route for landing page
@app.route("/")
def landing():
    return render_template("landing.html")  # Serve the landing page first

# Route for index page where the prediction form is
@app.route("/index")
def home():
    return render_template("index.html")  # This will be the page for predictions

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert categorical features to numerical values for the model
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    family_history_map = {"Yes": 1, "No": 0}
    benefits_map = {"Yes": 1, "No": 0}
    care_options_map = {"Yes": 1, "No": 0}
    anonymity_map = {"Yes": 1, "No": 0}
    leave_map = {"Easy": 1, "Difficult": 2, "Don't know": 0}

    # Convert work interference-related features into numerical values
    work_interfere_stress_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    work_interfere_workload_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    work_interfere_environment_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

    # Prepare the features for the model prediction
    work_interfere_stress = work_interfere_stress_map.get(data["work_interfere_stress"], -1)
    work_interfere_workload = work_interfere_workload_map.get(data["work_interfere_workload"], -1)
    work_interfere_environment = work_interfere_environment_map.get(data["work_interfere_environment"], -1)

    # If any work-related features are invalid, return an error response
    if work_interfere_stress == -1 or work_interfere_workload == -1 or work_interfere_environment == -1:
        return jsonify({"error": "Invalid input for work interference questions. Please provide valid responses."}), 400

    # Combine the three work interference questions into one
    # You can use different strategies to combine them, here we use an average
    work_interfere_combined = np.mean([work_interfere_stress, work_interfere_workload, work_interfere_environment])

    features = np.array([[ 
        data["age"],
        gender_map.get(data["gender"], -1),
        family_history_map.get(data["family_history"], -1),
        benefits_map.get(data["benefits"], -1),
        care_options_map.get(data["care_options"], -1),
        anonymity_map.get(data["anonymity"], -1),
        leave_map.get(data["leave"], -1),
        work_interfere_combined  # Use the combined feature here
    ]])

    # Make the prediction using the loaded model
    prediction = model.predict(features)[0]
    
    # Send the prediction back as a response
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

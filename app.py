import streamlit as st
import numpy as np
import tensorflow as tf

# Load the model
try:
    model = tf.keras.models.load_model("Fish_Spech_64.h5")
    print("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

fish_labels = ["Karpio", "Katla", "Shrimp", "Silver-Cap"]

st.title("Fish Species Prediction")

st.write("Enter the environmental parameters to predict the fish species.")

if model is not None:
    pH = st.number_input("Enter pH Value", min_value=0.0, max_value=14.0, value=7.0)
    temperature = st.number_input("Enter Temperature Value", min_value=-10.0, max_value=40.0, value=20.0)
    turbidity = st.number_input("Enter Turbidity Value", min_value=0.0, max_value=100.0, value=10.0)

    if st.button("Predict"):
        # Prepare input data for prediction
        # Assuming the model was trained on scaled data, we need to scale the user input as well.
        # We need the scaler object used during training. Since it's not available in the notebook state,
        # I'll assume for now that the model can handle unscaled data, but in a real application,
        # you would save and load the scaler as well.
        # For demonstration purposes, we'll use the values directly as input.
        # A more robust solution would involve saving and loading the scaler used previously.

        # Based on the previous cell where MinMax scaler was used:
        # ms=MinMaxScaler()
        # red_wine_data_X=ms.fit_transform(X)
        # We need to apply the same scaling to the new input.
        # Since the scaler object 'ms' is not available in the current session,
        # we'll need to re-create and fit it on the original data (or load it if saved).
        # For simplicity, let's re-fit the scaler on the original data 'X' from the notebook state.
        

        input_data = np.array([[(pH - 6) / (8.8 - 6),(temperature - 8) / (35 - 8),turbidity / 3]])


        #scaled_input_data = ms.transform(input_data)


        # Make predictions
        prediction = model.predict(input_data)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_species = fish_labels[predicted_class_index]

        st.success(f"Predicted Fish Species: {predicted_species}")
else:
    st.warning("Model is not loaded. Cannot make predictions.")

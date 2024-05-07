import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Predefined calorie information for common fruits and vegetables
from calories import calories_dict
from info import info_dict

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    with open("labels.txt") as f:
        content = f.readlines()
    label = [i[:-1] for i in content]
    return label[predicted_index]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if (app_mode == "Home"):
    st.header("FRUITS & VEGETABLES CALORIE & RECOGNITION SYSTEM ")
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project
elif (app_mode == "About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif (app_mode == "Prediction"):
    st.header("Model Prediction")
    option = st.radio("Choose Prediction Method:", ("Choose or Upload from Location", "Open Camera"))

    # If "Choose or Upload from Location" option is selected
    if option == "Choose or Upload from Location":
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
        if test_image is not None:
            image = Image.open(test_image)
            image_array = np.array(image)
            st.image(image, width=300, caption="Uploaded Image")
            if st.button("Predict", key="predict_button_upload"):
                # Convert image_array to a file-like object
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes.seek(0)

                predicted_label = model_prediction(image_bytes)
                with open("labels.txt") as f:
                    content = f.readlines()
                labels = [i.strip() for i in content]
                st.success("Model is Predicting it's a {}".format(predicted_label))

                # Display calorie information if available
                if predicted_label.lower() in calories_dict:
                    calorie_value = calories_dict[predicted_label.lower()]
                    st.info(f"The estimated calorie of the image is {calorie_value} calories per 100g")
                else:
                    st.info("Calorie information not available for {}".format(predicted_label))

                # Display basic information and health facts if available
                if predicted_label.lower() in info_dict:
                    info_text = info_dict[predicted_label.lower()]
                    st.subheader("Basic Information and Health Facts")
                    st.write(info_text)
                else:
                    st.info("Basic information and health facts not available for {}".format(predicted_label))

    # If "Open Camera" option is selected
    elif option == "Open Camera":
        stframe = st.empty()

        # Get a list of available webcam devices
        available_webcams = []
        for i in range(10):  # Check for up to 10 webcams
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_webcams.append(i)
                cap.release()
            except:
                continue

        # Allow the user to choose a webcam device
        selected_webcam = st.selectbox("Select Webcam", ["Default"] + [f"Webcam {i}" for i in available_webcams])

        # Initialize the camera
        if selected_webcam == "Default":
            cap = cv2.VideoCapture(0)
        else:
            webcam_index = int(selected_webcam.split(" ")[1])
            cap = cv2.VideoCapture(webcam_index)

        # Create a button for capturing and predicting
        predict_button = st.button("Capture and Predict", key="predict_button_camera")

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Show the frame in the Streamlit app
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Make a prediction if the button is clicked
            if predict_button:
                # Convert NumPy array (frame) to a file-like object
                image_bytes = io.BytesIO()
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pil_image.save(image_bytes, format='PNG')
                image_bytes.seek(0)

                predicted_label = model_prediction(image_bytes)
                with open("labels.txt") as f:
                        content = f.readlines()
                labels = [i.strip() for i in content]

                # Display the prediction
                st.success(f"Model is Predicting it's a {predicted_label}")

                # Display calorie information if available
                if predicted_label.lower() in calories_dict:
                    calorie_value = calories_dict[predicted_label.lower()]
                    st.info(f"The estimated calorie of the image is {calorie_value} calories per 100g")
                else:
                    st.info(f"Calorie information not available for {predicted_label}")

                # Display basic information and health facts if available
                if predicted_label.lower() in info_dict:
                    info_text = info_dict[predicted_label.lower()]
                    st.subheader("Basic Information and Health Facts")
                    st.write(info_text)
                else:
                    st.info(f"Basic information and health facts not available for {predicted_label}")

                # Reset the button state
                predict_button = False

            # Press 'q' to exit the camera loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("classification_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# Grad-CAM Function (unchanged from last fix)
def grad_cam(image_array, model, predicted_class):
    try:
        _ = model.predict(image_array)  # Initialize all layers
        last_conv_layer = None
        sequential_2 = None
        
        for layer in model.layers:
            if layer.name == "sequential_2":
                sequential_2 = layer
                break
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer

        if sequential_2 is not None:
            for sub_layer in sequential_2.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = sub_layer
        elif last_conv_layer is None:
            st.error("No Conv2D layer found in the model or within sequential_2.")
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None or tf.reduce_all(grads == 0):
            st.warning("Gradients are None or zero.")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Error in Grad-CAM: {str(e)}")
        return None

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .title {
        color: #2c3e50;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #e67e22;
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 10px;
    }
    .stTextInput>label, .stDateInput>label, .stNumberInput>label {
        color: #34495e;
        font-weight: bold;
        font-size: 16px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 14px;
    }
    .user-message {
        background-color: #dfe6e9;
        text-align: right;
    }
    .bot-message {
        background-color: #b3cde0;
        text-align: left;
    }
    .report-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
    }
    .report-header {
        color: #2c3e50;
        font-size: 28px;
        font-weight: bold;
        border-bottom: 2px solid #e67e22;
        padding-bottom: 10px;
    }
    .report-section {
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.5;
    }
    .report-label {
        color: #34495e;
        font-weight: bold;
        display: inline-block;
        width: 150px;
    }
    .report-value {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Chatbot
with st.sidebar:
    st.markdown('<p class="subtitle">Chat with Assistant</p>', unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    user_input = st.text_input("Ask about the report:", key="chat_input")
    if st.button("Send", key="send_chat"):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            context = st.session_state.report if "report" in st.session_state else "No report generated yet."
            chat_prompt = (
                f"You are a medical assistant named Dr. Alex Carter. The following is a generated report:\n\n{context}\n\n"
                f"Answer the user's question based on this report: {user_input}"
            )
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant named Dr. Alex Carter."},
                        {"role": "user", "content": chat_prompt},
                    ],
                    max_tokens=300,
                    temperature=0.7,
                )
                bot_response = response.choices[0].message.content.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.rerun()

# Main app layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">Brain Tumor Detection & Report Generator</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Patient Details", "Detection", "Report"])

# Tab 1: Patient Details
with tab1:
    st.markdown('<p class="subtitle">Enter Patient Details</p>', unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", key="patient_name")
    patient_id = st.text_input("Patient ID", key="patient_id")
    patient_age = st.number_input("Patient Age", min_value=0, max_value=150, value=0, key="patient_age")
    patient_date = st.date_input("Date of Scan", key="patient_date")
    
    if st.button("Save Details", key="save_details"):
        st.session_state.saved_patient_name = patient_name
        st.session_state.saved_patient_id = patient_id
        st.session_state.saved_patient_age = patient_age
        st.session_state.saved_patient_date = patient_date
        st.success("Patient details saved successfully!")

# Tab 2: Detection
with tab2:
    st.markdown('<p class="subtitle">Upload MRI Scan for Detection</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            # Preprocess the uploaded image
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image)
            st.write("Preprocessed image shape:", img_array.shape)

            # Load the model
            model = load_model()
            if model is None:
                raise ValueError("Model loading failed.")

            # Make prediction
            prediction = model.predict(img_array)
            # Updated class labels to match assumed model output order
            class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            st.write("Prediction probabilities:", prediction[0])
            
            predicted_class = np.argmax(prediction)
            confidence = prediction[0, predicted_class]
            result = f"Prediction: {class_labels[predicted_class]} (Confidence: {confidence:.2%})"
            
            # Store prediction results
            st.session_state.image = image
            st.session_state.img_array = img_array
            st.session_state.predicted_class = predicted_class
            st.session_state.class_labels = class_labels
            st.session_state.confidence = confidence

            # Generate heatmap
            heatmap = grad_cam(img_array, model, predicted_class)
            if heatmap is None or class_labels[predicted_class] == "No Tumor":
                superimposed_img = np.array(image.resize((224, 224)))
                heatmap_status = "No tumor detected; heatmap not applicable." if heatmap is not None else "Heatmap generation failed."
            else:
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(np.array(image.resize((224, 224))), 0.6, heatmap, 0.4, 0)
                heatmap_status = "Heatmap generated successfully."
            
            st.session_state.heatmap_status = heatmap_status
            st.session_state.superimposed_img = superimposed_img

            # Display Results
            st.image(image, caption=f'MRI Scan for {st.session_state.get("saved_patient_name", "Patient")}', use_container_width=True)
            st.write("### Result: ", result)
            st.write("Heatmap Status: ", heatmap_status)
            st.image(superimposed_img, caption='Image (with heatmap if applicable)', use_container_width=True)
            st.info("Switch to the 'Report' tab to generate the AI report.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Tab 3: Report
with tab3:
    st.markdown('<p class="subtitle">Generate AI Report</p>', unsafe_allow_html=True)
    if "predicted_class" not in st.session_state:
        st.warning("Please perform detection in the 'Detection' tab first.")
    else:
        if st.button("Generate Report", key="generate_report"):
            with st.spinner("Generating report..."):
                prompt = (
                    f"Generate a medical report based on the following information:\n\n"
                    f"Patient Name: {st.session_state.get('saved_patient_name', 'Not provided')}\n"
                    f"Patient ID: {st.session_state.get('saved_patient_id', 'Not provided')}\n"
                    f"Age: {st.session_state.get('saved_patient_age', 'Not provided') if st.session_state.get('saved_patient_age', 0) > 0 else 'Not provided'}\n"
                    f"Date of Scan: {st.session_state.get('saved_patient_date', 'Not provided')}\n"
                    f"Prediction: {st.session_state.class_labels[st.session_state.predicted_class]} (Confidence: {st.session_state.confidence:.2%})\n"
                    f"Heatmap Status: {st.session_state.heatmap_status}\n"
                    f"Prepared by: Dr. Alex Carter, Medical Assistant\n"
                    f"Please provide a detailed explanation suitable for a medical report."
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant named Dr. Alex Carter."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.7,
                )

                st.session_state.report = response.choices[0].message.content.strip()
        
        if "report" in st.session_state:
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown('<p class="report-header">Medical Report</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="report-section">', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Patient Name:</span> <span class="report-value">{st.session_state.get("saved_patient_name", "Not provided")}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Patient ID:</span> <span class="report-value">{st.session_state.get("saved_patient_id", "Not provided")}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Age:</span> <span class="report-value">{st.session_state.get("saved_patient_age", "Not provided") if st.session_state.get("saved_patient_age", 0) > 0 else "Not provided"}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Date of Scan:</span> <span class="report-value">{st.session_state.get("saved_patient_date", "Not provided")}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="report-section">', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Prediction:</span> <span class="report-value">{st.session_state.class_labels[st.session_state.predicted_class]}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Confidence:</span> <span class="report-value">{st.session_state.confidence:.2%}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="report-label">Heatmap Status:</span> <span class="report-value">{st.session_state.heatmap_status}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            st.markdown(f'<p class="report-value">{st.session_state.report}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.image(st.session_state.image, caption="Original MRI Scan", use_container_width=True)
            if st.session_state.heatmap_status == "Heatmap generated successfully.":
                st.image(st.session_state.superimposed_img, caption="Heatmap Overlay", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
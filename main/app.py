import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="PlantCare - Smart Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Load model info
@st.cache_data
def load_model_info():
    # Coba beberapa path yang mungkin
    base_dir = Path(__file__).parent.parent
    model_info_path = base_dir / "model" / "model_info.json"
    
    # Jika tidak ditemukan, coba path absolut
    if not model_info_path.exists():
        model_info_path = Path(r"D:\Download\plantvilage-sev\model\model_info.json")
    
    with open(model_info_path, 'r') as f:
        return json.load(f)

# Load model
@st.cache_resource
def load_model():
    # Path ke saved_model
    base_dir = Path(__file__).parent.parent
    saved_model_path = base_dir / "saved_model_format"
    
    # Jika tidak ditemukan, coba path absolut
    if not saved_model_path.exists():
        saved_model_path = Path(r"D:\Download\plantvilage-sev\saved_model_format")
    
    if not saved_model_path.exists():
        raise FileNotFoundError(f"Model not found at {saved_model_path}")
    
    # Load SavedModel menggunakan tf.saved_model.load
    loaded_model = tf.saved_model.load(str(saved_model_path))
    return loaded_model

# Preprocessing image
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image untuk prediksi"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediksi
def predict(image, model, class_names):
    """Melakukan prediksi pada gambar"""
    processed_img = preprocess_image(image)
    
    # Untuk tf.saved_model.load format
    if hasattr(model, 'signatures'):
        # Ambil serving signature
        infer = model.signatures['serving_default']
        
        # Cek input signature untuk mengetahui tipe data yang diharapkan
        input_name = list(infer.structured_input_signature[1].keys())[0]
        input_spec = infer.structured_input_signature[1][input_name]
        
        # Buat input tensor sesuai dengan dtype yang diharapkan
        if input_spec.dtype == tf.float16:
            input_tensor = tf.constant(processed_img, dtype=tf.float16)
        else:
            input_tensor = tf.constant(processed_img, dtype=tf.float32)
        
        # Get prediction
        output_dict = infer(**{input_name: input_tensor})
        output_key = list(output_dict.keys())[0]
        predictions = output_dict[output_key].numpy()[0]
    else:
        # Regular Keras model (fallback)
        predictions = model.predict(processed_img, verbose=0)[0]
    
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    # Ambil top 3 prediksi
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3_predictions = [
        {
            "class": class_names[idx],
            "confidence": float(predictions[idx])
        }
        for idx in top_3_idx
    ]
    
    return {
        "predicted_class": class_names[predicted_class_idx],
        "confidence": float(confidence),
        "top_3": top_3_predictions
    }

# Format nama kelas untuk ditampilkan
def format_class_name(class_name):
    """Format nama kelas agar lebih readable"""
    parts = class_name.replace("___", " - ").replace("_", " ")
    return parts

# Load model dan info
try:
    model_info = load_model_info()
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Header
st.markdown("""
    <div class="header">
        <h1>🌾 PlantCare </h1>
        <p>Smart Agricultural Disease Detection for Modern Farmers</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown(f"**Architecture:** {model_info['architecture']}")
    st.markdown(f"**Training Accuracy:** {model_info['training_accuracy']*100:.2f}%")
    st.markdown(f"**Validation Accuracy:** {model_info['validation_accuracy']*100:.2f}%")
    st.markdown(f"**Model Size:** {model_info['model_size_mb']:.2f} MB")
    st.markdown(f"**Classes:** {model_info['num_classes']}")
    
    st.markdown("---")
    st.markdown("### 🌱 Supported Plants")
    
    # Ekstrak nama tanaman unik
    plants = sorted(set([name.split("___")[0] for name in model_info['class_names']]))
    for plant in plants:
        st.markdown(f"• {plant.replace('_', ' ')}")
    
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Upload an image of a plant leaf
    2. Wait for the model to analyze
    3. View the prediction results
    4. Check the confidence score
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Tombol prediksi
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                result = predict(image, model, model_info['class_names'])
                st.session_state['result'] = result
                st.session_state['analyzed'] = True

with col2:
    st.markdown("### 📋 Analysis Results")
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        result = st.session_state['result']
        
        # Parse hasil
        plant, condition = result['predicted_class'].split("___")
        plant_name = plant.replace("_", " ")
        condition_name = condition.replace("_", " ")
        
        # Status card
        if "healthy" in condition.lower():
            status_color = "#28a745"
            status_icon = "✅"
            status_text = "Healthy"
        else:
            status_color = "#dc3545"
            status_icon = "⚠️"
            status_text = "Disease Detected"
        
        st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid {status_color};">
                <h3>{status_icon} {status_text}</h3>
                <p><strong>Plant:</strong> {plant_name}</p>
                <p><strong>Condition:</strong> {condition_name}</p>
                <p><strong>Confidence:</strong> {result['confidence']*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress bar untuk confidence
        st.progress(result['confidence'])
        
        # Top 3 predictions
        st.markdown("#### 📊 Top 3 Predictions")
        for i, pred in enumerate(result['top_3'], 1):
            formatted_name = format_class_name(pred['class'])
            confidence_pct = pred['confidence'] * 100
            
            st.markdown(f"""
                <div class="prediction-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{i}.</strong> {formatted_name}</span>
                        <span style="color: #666;">{confidence_pct:.2f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Rekomendasi berdasarkan hasil
        st.markdown("#### 💡 Recommendations")
        if "healthy" in condition.lower():
            st.success("✅ Your plant appears healthy! Continue regular care and monitoring.")
        else:
            st.warning(f"⚠️ Disease detected: **{condition_name}**")
            st.info("""
            **Recommended Actions:**
            - Isolate affected plants if possible
            - Remove infected leaves
            - Consult with a local agricultural expert
            - Consider appropriate treatment methods
            - Monitor other plants for similar symptoms
            """)
    else:
        st.info("👆 Upload an image and click 'Analyze Image' to see results")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌾 PlantCare Pro - Powered by Deep Learning | ResNet50 Architecture</p>
        <p style="font-size: 0.9em;">Empowering farmers with AI-driven plant health monitoring</p>
    </div>
""", unsafe_allow_html=True)

# 🌿 AgriScan - Plant Disease Detection System

Sistem deteksi penyakit tanaman berbasis Deep Learning menggunakan MobileNetV2 dan Streamlit.

## 📂 Struktur Project

```
plantvilage-sev/
│
├── main/                  # Aplikasi Streamlit
│   ├── app.py            # Main application
│   ├── style.css         # Custom styling
│   ├── requirements.txt  # Dependencies
│   └── README.md         # Dokumentasi lengkap
│
└── saved_model/          # TensorFlow SavedModel
    ├── saved_model.pb
    └── variables/
```

## 🚀 Quick Start

### Menggunakan Virtual Environment (Disarankan)

```bash
# Masuk ke folder main
cd main

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

### Tanpa Virtual Environment

```bash
# Masuk ke folder main
cd main

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

> **💡 Tips**: Gunakan virtual environment untuk menghindari konflik dependencies dengan project Python lainnya.

## 📖 Dokumentasi Lengkap

Untuk dokumentasi lengkap, lihat [main/README.md](main/README.md)

## ✨ Fitur

- ✅ Deteksi 38 kondisi dari 14 jenis tanaman
- ✅ UI modern dan responsive
- ✅ Real-time inference
- ✅ Top 3 predictions
- ✅ Smart recommendations
- ✅ Confidence scoring

## 🌱 Supported Plants

Apple • Blueberry • Cherry • Corn • Grape • Orange • Peach • Pepper • Potato • Raspberry • Soybean • Squash • Strawberry • Tomato

## 📊 Model Performance

- **Training Accuracy**: 96.16%
- **Validation Accuracy**: 88.95%
- **Model Size**: 1.62 MB
- **Architecture**: MobileNetV2

## 🛠️ Tech Stack

- **Backend/Frontend**: Streamlit
- **Deep Learning**: TensorFlow 2.16+
- **Model**: MobileNetV2 Transfer Learning
- **Image Processing**: Pillow
- **Python**: 3.10+

## 📝 License

Educational & Research Purpose

---


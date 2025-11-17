# 🌿 AgriScan - Plant Disease Classifier

Aplikasi web berbasis Streamlit untuk deteksi penyakit tanaman menggunakan Deep Learning. Sistem ini dapat mengidentifikasi 38 jenis kondisi tanaman (sehat dan penyakit) dari 14 jenis tanaman berbeda.

![AgriScan](https://img.shields.io/badge/Model-MobileNetV2-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)

## 📋 Deskripsi

AgriScan adalah sistem deteksi penyakit tanaman yang menggunakan model MobileNetV2 dengan Transfer Learning. Aplikasi ini dirancang dengan arsitektur monolitik (tanpa pemisahan frontend-backend) menggunakan Streamlit untuk kemudahan deployment dan penggunaan.

### ✨ Fitur Utama

- 🖼️ **Upload & Analisis**: Upload gambar daun tanaman dan dapatkan hasil analisis instant
- 📊 **Confidence Score**: Tampilan persentase tingkat kepercayaan prediksi
- 🏆 **Top 3 Predictions**: Lihat 3 kemungkinan hasil teratas
- 💡 **Smart Recommendations**: Saran otomatis berdasarkan hasil deteksi
- 🎨 **UI Modern**: Interface yang clean, responsive, dan user-friendly
- ⚡ **Fast Inference**: Prediksi cepat menggunakan model yang dioptimasi

### 🌱 Tanaman yang Didukung

Sistem dapat mendeteksi kondisi dari 14 jenis tanaman:

- 🍎 Apple (Apel)
- 🫐 Blueberry
- 🍒 Cherry
- 🌽 Corn (Jagung)
- 🍇 Grape (Anggur)
- 🍊 Orange (Jeruk)
- 🍑 Peach (Persik)
- 🌶️ Pepper (Paprika)
- 🥔 Potato (Kentang)
- 🍓 Raspberry
- 🫘 Soybean (Kedelai)
- 🎃 Squash (Labu)
- 🍓 Strawberry (Stroberi)
- 🍅 Tomato (Tomat)

### 🔬 Model Information

- **Architecture**: MobileNetV2 with Transfer Learning
- **Input Size**: 224 x 224 x 3
- **Number of Classes**: 38 (termasuk kondisi sehat dan berbagai penyakit)
- **Training Accuracy**: 96.16%
- **Validation Accuracy**: 88.95%
- **Model Format**: TensorFlow SavedModel
- **Model Size**: ~1.62 MB

## 📁 Struktur Project

```
plantvilage-sev/
├── main/
│   ├── app.py              # Aplikasi Streamlit utama
│   ├── style.css           # Custom CSS styling
│   ├── requirements.txt    # Python dependencies
│   ├── README.md          # Dokumentasi (file ini)
│   └── venv/              # Virtual environment (optional)
│
└── saved_model/           # Model TensorFlow SavedModel
    ├── saved_model.pb
    ├── variables/
    └── fingerprint.pb
```

## 🚀 Instalasi & Setup

### Prerequisites

- Python 3.10 atau lebih tinggi
- pip (Python package manager)
- 4GB+ RAM (untuk running model)

### Langkah Instalasi

1. **Clone atau Download Project**
   ```bash
   cd c:\project\plantvilage-sev\main
   ```

2. **Buat Virtual Environment (Opsional tapi Disarankan)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies yang akan diinstall:
   - streamlit (>=1.31.0) - Web framework
   - tensorflow (>=2.16.1) - Deep learning framework
   - pillow (>=10.0.0) - Image processing
   - numpy (>=1.26.0) - Numerical computing

## 💻 Cara Menjalankan

### Method 1: Standard Run

```bash
streamlit run app.py
```

### Method 2: Auto-reload on Save

```bash
streamlit run app.py --server.runOnSave true
```

### Method 3: Custom Port

```bash
streamlit run app.py --server.port 8080
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

## 📖 Cara Menggunakan

1. **Upload Gambar**
   - Klik tombol "Browse files" atau drag & drop
   - Format yang didukung: JPG, JPEG, PNG
   - Pastikan gambar menampilkan daun tanaman dengan jelas

2. **Analisis**
   - Klik tombol **"🔍 Analyze Image"**
   - Tunggu beberapa detik untuk proses prediksi

3. **Lihat Hasil**
   - **Status**: Healthy ✅ atau Disease Detected ⚠️
   - **Plant**: Nama tanaman yang terdeteksi
   - **Condition**: Kondisi atau jenis penyakit
   - **Confidence**: Persentase tingkat kepercayaan (0-100%)
   - **Top 3 Predictions**: Alternatif kemungkinan lainnya

4. **Baca Rekomendasi**
   - Jika sehat: Tips perawatan lanjutan
   - Jika sakit: Langkah-langkah penanganan

## 🎨 Kustomisasi

### Mengubah Styling

Edit file `style.css` untuk mengubah tampilan:

```css
/* Contoh: Ubah warna gradient header */
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Mengubah Port Default

Buat file `.streamlit/config.toml`:

```toml
[server]
port = 8080
headless = true
```

### Menambah Fitur

Edit `app.py` dan tambahkan fungsi baru sesuai kebutuhan.

## 🔧 Troubleshooting

### Error: Model not found

**Solusi**: Pastikan folder `saved_model` ada di lokasi yang benar:
```
c:\project\plantvilage-sev\saved_model\
```

### Error: Module not found

**Solusi**: Install ulang dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Aplikasi Lambat

**Solusi**: 
- Pastikan GPU tersedia (jika ada)
- Reduce image size sebelum upload
- Clear cache Streamlit (tekan `C` di browser)

### Port Already in Use

**Solusi**: Gunakan port berbeda:
```bash
streamlit run app.py --server.port 8502
```

## 📊 Performa

- **Inference Time**: ~1-3 detik (CPU)
- **Inference Time**: ~0.5-1 detik (GPU)
- **Memory Usage**: ~500MB-1GB
- **Supported Browsers**: Chrome, Firefox, Safari, Edge

## 🛠️ Development

### Running in Development Mode

```bash
streamlit run app.py --server.runOnSave true --server.fileWatcherType auto
```

### Testing

Upload berbagai gambar dari dataset PlantVillage atau gambar real-world untuk testing.

## 📝 Model Classes

Aplikasi dapat mendeteksi 38 kondisi berbeda:

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry___healthy
7. Cherry___Powdery_mildew
8. Corn___Cercospora_leaf_spot Gray_leaf_spot
9. Corn___Common_rust
10. Corn___healthy
... (dan 28 kelas lainnya)

*Lihat `model_info.json` untuk daftar lengkap*

## 🤝 Contributing

Jika ingin berkontribusi:
1. Fork repository
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Project ini dibuat untuk tujuan edukasi dan penelitian.

## 👥 Authors

AgriScan Development Team

## 🙏 Acknowledgments

- Dataset: PlantVillage Dataset
- Model Architecture: MobileNetV2 (Google)
- Framework: TensorFlow & Streamlit
- UI Inspiration: Modern web design principles

## 📞 Support

Jika ada pertanyaan atau masalah:
- Open an issue di GitHub
- Contact: [email/contact info]

---

<div align="center">
  
**🌿 AgriScan - Membantu Petani Mendeteksi Penyakit Tanaman Sejak Dini**

Made with ❤️ using TensorFlow & Streamlit

</div>


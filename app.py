import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- KONFIGURASI ---
MODEL_PATH = 'model_buah_final.h5'
LABEL_PATH = 'labels.txt'
IMG_SIZE = (224, 224)
THRESHOLD = 0.7  # Akurasi minimal 70% baru ditampilkan

# 1. Load Model & Label
print("Sedang memuat model...")
model = load_model(MODEL_PATH)
with open(LABEL_PATH, 'r') as f:
    class_names = f.read().splitlines()
print(f"Model Siap! Mendeteksi: {class_names}")

# 2. Buka Kamera
cap = cv2.VideoCapture(0) # Ubah ke 1 jika menggunakan kamera eksternal

if not cap.isOpened():
    print("Kamera tidak terdeteksi!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break

    # Simpan frame asli untuk ditampilkan
    display_frame = frame.copy()

    # --- PREPROCESSING GAMBAR ---
    # Resize ke 224x224 (Wajib untuk MobileNetV2)
    img = cv2.resize(frame, IMG_SIZE)
    # Ubah warna BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Tambah dimensi batch (menjadi 1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    # Preprocessing khusus MobileNetV2
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # --- PREDIKSI ---
    predictions = model.predict(img, verbose=0)
    score = np.max(predictions)               # Ambil nilai probabilitas tertinggi
    class_index = np.argmax(predictions)      # Ambil index kelasnya
    label_name = class_names[class_index]     # Ambil nama buahnya

    # --- VISUALISASI ---
    h, w, _ = display_frame.shape
    
    # Tentukan warna kotak (Hijau jika yakin, Merah jika ragu)
    if score > THRESHOLD:
        box_color = (0, 255, 0) # Hijau
        label_text = f"{label_name}: {score*100:.1f}%"
    else:
        box_color = (0, 0, 255) # Merah
        label_text = "Tidak Yakin / Bukan Buah"

    # Gambar kotak fokus di tengah layar
    center_x, center_y = w // 2, h // 2
    box_size = 250
    x1, y1 = center_x - box_size // 2, center_y - box_size // 2
    x2, y2 = center_x + box_size // 2, center_y + box_size // 2
    
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # Tampilkan layar
    cv2.imshow('Deteksi Buah Real-Time', display_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# --- CONFIG ---
class_names = ['Mangga', 'apel', 'jeruk', 'pisang', 'strawberry']
model = load_model('model_buah_final2.h5')
print("Model loaded.")

camera = cv2.VideoCapture(0)

# Global Variables
last_detected_info = None  
last_capture_time = 0      
CONFIDENCE_THRESHOLD = 60.0 # Batas minimum akurasi untuk dianggap "Ada Object"

def generate_frames():
    global last_detected_info, last_capture_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 1. Resize & Preprocessing untuk AI
            img_small = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            img_array = image.img_to_array(img_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array) 
            
            # 2. Prediksi
            prediction = model.predict(img_array, verbose=0)
            index_max = np.argmax(prediction)
            class_result = class_names[index_max]
            confidence = np.max(prediction) * 100
            
            # --- LOGIKA UI (Kotak & Teks) ---
            
            # Hanya tampilkan jika confidence di atas batas (misal 60%)
            if confidence > CONFIDENCE_THRESHOLD:
                
                # --- A. Cari Lokasi Benda (Trick Computer Vision) ---
                # Karena model klasifikasi tidak memberi koordinat, kita cari manual pake Contours
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                # Deteksi tepi (Canny Edge Detection)
                edged = cv2.Canny(blur, 50, 150)
                # Cari kontur
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Jika ada benda yang terdeteksi secara visual
                if contours:
                    # Ambil kontur terbesar (diasumsikan itu buahnya)
                    c = max(contours, key=cv2.contourArea)
                    
                    # Hanya gambar kotak jika bendanya lumayan besar (filter noise)
                    if cv2.contourArea(c) > 2000:
                        x, y, w, h = cv2.boundingRect(c)
                        
                        # Gambar Kotak Hijau Pas di Benda
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Tulis Label di atas kotak
                        label_text = f"{class_result}: {confidence:.1f}%"
                        # Background hitam kecil buat teks biar terbaca
                        cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 255, 0), -1) 
                        cv2.putText(frame, label_text, (x + 5, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # --- B. Logic Auto Capture ---
                # Tetap jalan jika akurasi > 70% (bisa beda dengan threshold tampilan)
                current_time = time.time()
                if confidence > 70 and (current_time - last_capture_time) > 3:
                    _, buffer_jpg = cv2.imencode('.jpg', frame)
                    jpg_as_text = base64.b64encode(buffer_jpg).decode('utf-8')
                    
                    last_detected_info = {
                        'class': class_result,
                        'confidence': f"{confidence:.2f}",
                        'image': jpg_as_text,
                        'timestamp': time.strftime("%H:%M:%S")
                    }
                    last_capture_time = current_time
                    print(f"Auto-Captured: {class_result} ({confidence:.2f}%)")
            
            # Jika Confidence < 60%, Loop akan lewat begitu saja 
            # (Frame tetap bersih, tidak ada kotak, tidak ada teks)

            # Encode frame video untuk dikirim ke web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTES ---

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/kamera')
def camera_page():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_capture_data')
def get_capture_data():
    global last_detected_info
    if last_detected_info:
        data = last_detected_info
        last_detected_info = None 
        return jsonify({'status': 'new', 'data': data})
    else:
        return jsonify({'status': 'empty'})

if __name__ == '__main__':
    app.run(debug=True)
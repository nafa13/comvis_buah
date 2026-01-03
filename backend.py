import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- CONFIG ---
class_names = ['Apel', 'Jeruk', 'Mangga', 'Pisang']
model = load_model('model_buah_final.h5')
print("Model loaded.")

camera = cv2.VideoCapture(0)

# Global Variables untuk fitur Auto-Capture
last_detected_info = None  # Menyimpan info: {'class': 'Apel', 'conf': 90.5, 'image': 'base64str...'}
last_capture_time = 0      # Cooldown agar tidak capture 100x per detik

def generate_frames():
    global last_detected_info, last_capture_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize untuk model
            img_small = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            # Preprocessing
            img_array = image.img_to_array(img_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0 
            
            # Predict
            prediction = model.predict(img_array, verbose=0)
            index_max = np.argmax(prediction)
            class_result = class_names[index_max]
            confidence = np.max(prediction) * 100
            
            # --- LOGIC AUTO-CAPTURE ---
            current_time = time.time()
            # Capture jika akurasi > 70% DAN sudah lewat 3 detik sejak capture terakhir
            if confidence > 70 and (current_time - last_capture_time) > 3:
                
                # Encode gambar frame saat ini ke Base64 agar bisa dikirim ke HTML tanpa simpan file
                _, buffer_jpg = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer_jpg).decode('utf-8')
                
                last_detected_info = {
                    'class': class_result,
                    'confidence': f"{confidence:.2f}",
                    'image': jpg_as_text,
                    'timestamp': time.strftime("%H:%M:%S")
                }
                last_capture_time = current_time # Reset cooldown
                print(f"Auto-Captured: {class_result} ({confidence:.2f}%)")

            # Visualisasi di Video
            color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
            # Kotak transparan di atas
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            text = f"{class_result}: {confidence:.1f}%"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)
            
            # Encode frame video
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

# API Baru: Untuk memberikan data capture ke halaman dashboard/kamera
@app.route('/get_capture_data')
def get_capture_data():
    global last_detected_info
    if last_detected_info:
        # Kirim data dan kemudian kosongkan agar tidak dikirim ulang terus menerus
        data = last_detected_info
        last_detected_info = None 
        return jsonify({'status': 'new', 'data': data})
    else:
        return jsonify({'status': 'empty'})

if __name__ == '__main__':
    app.run(debug=True)
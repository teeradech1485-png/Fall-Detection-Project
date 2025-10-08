import cv2
import datetime
import requests
import threading
from flask import Flask, Response
from pyngrok import ngrok
from ultralytics import YOLO

# --- 1. ตั้งค่าพื้นฐาน ---
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1336548845366280192/EWhYThzReXkCtjmQL8E5f5IosXsjc3_a8iz6cAe2vzkvpc0SOjE9EvW_jH28xvE9BSPc'
NGROK_AUTHTOKEN = "32Uz6XpcNrUdL3e5xxe9IkgyqAn_R2daSUgAGf89rpcVpbcw"
model = YOLO("best.pt")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

# สร้าง Flask app และตัวแปรที่ใช้ร่วมกัน
app = Flask(__name__)
output_frame = None
lock = threading.Lock()
falling_detected = False

# --- 2. ฟังก์ชันหลักในการประมวลผลวิดีโอและตรวจจับ ---
def process_video_feed():
    global output_frame, falling_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ประมวลผลภาพด้วยโมเดล YOLO โดยปรับขนาดเป็น 416x224 (height, width)
        results = model(frame, imgsz=(224, 416)) 

        # วาดกรอบผลลัพธ์ลงบนภาพ (ไลบรารีจะวาดบน 'frame' เดิมให้)
        annotated_frame = results[0].plot()

        # ตรวจสอบผลลัพธ์เพื่อหาการล้ม
        found_falling_this_frame = False
        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name.lower() in ["fall-down", "falling"]:
                    found_falling_this_frame = True
                    break
        
        # จัดการการแจ้งเตือนเมื่อพบการล้ม
        if found_falling_this_frame and not falling_detected:
            print("🚨 ตรวจพบการล้ม! กำลังส่งการแจ้งเตือน...")
            falling_detected = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"fall_{timestamp}.jpg"
            cv2.imwrite(image_path, annotated_frame)
            send_image_to_discord(image_path, "🚨 ตรวจพบการล้ม!")
        elif not found_falling_this_frame:
            falling_detected = False

        # อัปเดตเฟรมสำหรับสตรีม
        with lock:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            output_frame = buffer.tobytes()

        # แสดงหน้าต่างภาพบนโน้ตบุ๊ก
        cv2.imshow("Live Fall Detection (Local Preview)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ปิดการทำงานของกล้อง...")
            break
            
    cap.release()
    cv2.destroyAllWindows()

def send_image_to_discord(image_path, message):
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'content': message}
            response = requests.post(DISCORD_WEBHOOK_URL, files=files, data=data)
            if response.status_code == 204 or response.status_code == 200:
                print("✅ ส่งการแจ้งเตือนไปที่ Discord สำเร็จแล้ว")
            else:
                print(f"⚠️ ไม่สามารถส่งการแจ้งเตือนได้: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการส่งไป Discord: {e}")

# --- 3. ฟังก์ชันสำหรับการสตรีมวิดีโอผ่าน Flask ---
def generate_stream():
    global output_frame
    while True:
        if not cap.isOpened():
             break
        with lock:
            if output_frame is None:
                continue
            frame = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <head><title>CCTV-FallGuard</title>
            <style>
                body { font-family: sans-serif; text-align: center; background-color: #2c3e50; color: white; }
                img { max-width: 90%; margin-top: 20px; border: 3px solid #3498db; border-radius: 8px; }
            </style>
        </head>
        <body>
            <h1>🎥 CCTV-FallGuard</h1>
            <img src="/video">
        </body>
    </html>
    """

# --- 4. ส่วนสำหรับรันโปรแกรม ---
if __name__ == '__main__':
    processing_thread = threading.Thread(target=process_video_feed)
    processing_thread.daemon = True
    processing_thread.start()

    try:
        ngrok.set_auth_token(NGROK_AUTHTOKEN)
        public_url = ngrok.connect(5000)
        print("=====================================================")
        print(f"🌍 สามารถเปิดดูกล้องผ่านเว็บได้ที่: {public_url}")
        print("💻 หน้าต่างแสดงผลบนเครื่องจะเปิดขึ้นมาอัตโนมัติ")
        print("🔴 กดปุ่ม 'q' บนหน้าต่างกล้องเพื่อปิดการทำงาน")
        print("=====================================================")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ ngrok ได้: {e}")
        exit()
        
    app.run(host="0.0.0.0", port=5000)
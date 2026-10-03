"""
Helper script to prepare YOLOv8 model weights in weights/yolov8n.onnx.
"""
import os
import shutil

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "yolov8n.onnx")

def prepare_yolo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        print(f"[*] File weights '{OUTPUT_FILE}' sudah tersedia.")
        return

    print("[*] Menyiapkan model YOLOv8...")
    try:
        from ultralytics import YOLO
        print("[*] Mengekspor model YOLOv8n ke format ONNX...")
        model = YOLO("yolov8n.pt")
        exported_path = model.export(format="onnx")
        if exported_path and os.path.exists(exported_path):
            shutil.move(exported_path, OUTPUT_FILE)
            print(f"[✓] Berhasil menyimpan model ONNX ke: {OUTPUT_FILE}")
    except Exception as e:
        print(f"[*] Catatan: Model YOLO ONNX belum dibuat ({e}).")
        print("    Sistem akan otomatis menggunakan detektor OpenCV Haar Cascade bawaan.")
        print("    (Untuk mengaktifkan YOLO 80 kelas: pip install ultralytics && yolo export model=yolov8n.pt format=onnx)")

if __name__ == "__main__":
    prepare_yolo()

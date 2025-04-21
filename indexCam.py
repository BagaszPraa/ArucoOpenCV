import cv2

for index in range(5):  # coba dari index 0 sampai 4
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"[✓] Kamera ditemukan di index {index}")
        cap.release()
    else:
        print(f"[X] Tidak ada kamera di index {index}")

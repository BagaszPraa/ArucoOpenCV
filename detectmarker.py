import cv2
import numpy as np

# === Load hasil kalibrasi ===
data = np.load("calibration_data.npz")
camera_matrix = data["cameraMatrix"]
dist_coeffs = data["distCoeffs"]

# === Setup ArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Ukuran sisi marker dalam meter (misal 5 cm)
marker_length = 0.05

# === Buka kamera ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # Gambar kotak di sekitar marker
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimasi pose marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

    cv2.imshow("ArUco Pose Estimation", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import math

# === Load hasil kalibrasi kamera ===
data = np.load("calibration_data.npz")
camera_matrix = data["cameraMatrix"]
dist_coeffs = data["distCoeffs"]

# Setup ArUco marker detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
parameters = cv2.aruco.DetectorParameters()

marker_mm = 31.6  # Marker size in mm
marker_length = marker_mm / 1000
# Initialize video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Fungsi untuk menghitung pose kamera relatif terhadap marker
def get_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)  # Konversi rotasi vektor ke matriks rotasi
    R_inv = R.T  # Transpose matriks rotasi
    tvec_inv = -R_inv @ tvec.reshape(3, 1)  # Transformasi translasi
    return R_inv, tvec_inv

# Fungsi untuk mengonversi matriks rotasi ke sudut Euler
def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

# Loop utama untuk mendeteksi marker dan menghitung pose
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi marker pada frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Gambar marker yang terdeteksi
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimasi pose marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            # Gambar sumbu pose pada marker
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Hitung posisi dan orientasi kamera
            R_cam, t_cam = get_camera_pose(rvec, tvec)
            x, y, z = t_cam.flatten()
            roll, pitch, yaw = rotation_matrix_to_euler_angles(R_cam)

            # Tampilkan data posisi dan orientasi di konsol
            print(f"X = {x:.2f} || Y = {y:.2f} || Z = {z:.2f} || ROLL = {roll:.1f} || PIT = {pitch:.1f} || YAW = {yaw:.1f}")

            # Tambahkan teks posisi dan orientasi pada frame
            cv2.putText(frame, f"Pos: x={x:.2f} y={y:.2f} z={z:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rot: roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Tampilkan frame dengan overlay
    cv2.imshow("ArUco Pose - Precision Landing", frame)

    # Tekan 'Esc' untuk keluar
    if cv2.waitKey(1) == 27:
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()

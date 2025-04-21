import cv2
import numpy as np
import math

# === Load hasil kalibrasi kamera ===
data = np.load("c922_dataCalib.npz")
camera_matrix = data["cameraMatrix"]
dist_coeffs = data["distCoeffs"]

# Setup ArUco marker detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Dictionary untuk ukuran marker (ID marker -> ukuran dalam meter)
marker_sizes = {
    # 0: 155 / 1000,  # Marker ID 0 dengan ukuran 155 mm (15.5 cm)
    # 1: 47.2 / 1000,  # Marker ID 1 dengan ukuran 47.2 mm (4.72 cm)
    2: 31.6 / 1000,  # Marker ID 2 dengan ukuran 31.6 mm (3.16 cm)
    # 4: 92.4 / 1000,  # Marker ID 4 dengan ukuran 92.4 mm (9.24 cm)
    # Tambahkan ukuran marker lainnya di sini
}

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

        # Iterasi melalui setiap marker yang terdeteksi
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_sizes:
                marker_length = marker_sizes[marker_id]  # Ambil ukuran marker berdasarkan ID
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], marker_length, camera_matrix, dist_coeffs
                )

                rvec, tvec = rvecs[0], tvecs[0]

                # Gambar sumbu pose pada marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                # Hitung posisi dan orientasi kamera
                R_cam, t_cam = get_camera_pose(rvec, tvec)
                x, y, z = t_cam.flatten()
                roll, pitch, yaw = rotation_matrix_to_euler_angles(R_cam)

                # Tampilkan data posisi dan orientasi di konsol
                # print(f"Marker ID: {marker_id} || X = {x:.2f} || Y = {y:.2f} || Z = {z:.2f} || ROLL = {roll:.1f} || PIT = {pitch:.1f} || YAW = {yaw:.1f}")

                # Tambahkan teks posisi dan orientasi pada frame
                cv2.putText(frame, f"ID: {marker_id} Pos: x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw:.1f}",
                            (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f"Rot: roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}",
                #             (10, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Tampilkan frame dengan overlay
    cv2.imshow("ArUco Pose - Precision Landing", frame)

    # Tekan 'Esc' untuk keluar
    if cv2.waitKey(1) == 27:
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
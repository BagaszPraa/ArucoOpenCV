import cv2
import numpy as np
import math

# === Load hasil kalibrasi ===
data = np.load("calibration_data.npz")
camera_matrix = data["cameraMatrix"]
dist_coeffs = data["distCoeffs"]

# Setup ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
parameters = cv2.aruco.DetectorParameters()

marker_length = 0.05  # meter

# cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def get_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    tvec_inv = -R_inv @ tvec.reshape(3, 1)
    return R_inv, tvec_inv

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            R_cam, t_cam = get_camera_pose(rvec, tvec)
            x, y, z = t_cam.flatten()
            # cv2.putText(frame, f"Pos: x={x:.2f} y={y:.2f} z={z:.2f}",
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            roll, pitch, yaw = rotation_matrix_to_euler_angles(R_cam)
            # cv2.putText(frame, f"Rot: roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}",
            #             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            print(f"X = {x:.2f} || Y = {y:.2f} || Z = {z:.2f} || ROLL = {roll:.1f} || PIT = {pitch:.1f} || YAW = {yaw:.1f}")
            # print(f"Rotation: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")

    cv2.imshow("ArUco Pose - OpenCV 3.9", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

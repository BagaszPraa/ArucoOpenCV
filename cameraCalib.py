import cv2
import numpy as np

# Ukuran checkerboard (jumlah sudut dalam baris & kolom, bukan jumlah kotak)
CHECKERBOARD = (9, 6)  # artinya 10x7 kotak
SQUARE_SIZE = 0.025  # ukuran sisi kotak dalam meter (misalnya 2.5 cm)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Persiapkan object points seperti (0,0,0), (1,0,0), ..., dalam satuan meter
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # Titik 3D di dunia nyata
imgpoints = []  # Titik 2D di image

cap = cv2.VideoCapture(0)

print("🔧 Tekan [SPACE] untuk ambil gambar, [ESC] untuk kalibrasi dan keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)

    cv2.imshow("Kalibrasi Kamera", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32 and found:  # SPACE
        print("✅ Gambar ditambahkan")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("❌ Tidak cukup gambar untuk kalibrasi. Minimal 5.")
    exit()

print("🧮 Mengkalibrasi kamera...")

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n📷 Kalibrasi selesai!")
print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coefficients:\n", distCoeffs)

# Hitung rata-rata error proyeksi ulang
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print("🔍 Reprojection error rata-rata: {:.4f}".format(total_error / len(objpoints)))

# Simpan hasil kalibrasi
np.savez("calibration_data.npz", cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
print("\n💾 Data disimpan ke calibration_data.npz")

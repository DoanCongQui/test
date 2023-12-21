import cv2
import nanocamera as nano

camera = nano.Camera(flip=0, width=1280, height=800, fps=30)
frame = camera.read()

if frame is not None:
    kernel_size = (3, 3)
    gauss_image = cv2.GaussianBlur(frame, kernel_size, 0)
    # Tiếp tục xử lý hình ảnh của bạn ở đây
else:
    print("Không thể đọc được hình ảnh từ camera.")

import cv2
import numpy as np


def process_frame(frame):
    # Chuyển đổi sang không gian màu HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Xác định vùng màu của làn đường (ví dụ: màu trắng)
    lower_lane_color = np.array([0, 0, 200])
    upper_lane_color = np.array([255, 50, 255])
    lane_mask = cv2.inRange(hsv_image, lower_lane_color, upper_lane_color)

    # Phát hiện biên cạnh
    edges = cv2.Canny(lane_mask, 50, 150)

    # Giảm nhiễu
    blurred_edges = cv2.GaussianBlur(edges, (5, 5), 0)

    # Xác định vùng quan tâm
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi_edges = region_of_interest(blurred_edges, vertices)

    # Phân tích đường lái xe (Hough Line Transform)
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Hiển thị đường lái xe lên ảnh gốc
    result = display_lines(frame, lines)

    return result


def region_of_interest(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    roi_edges = cv2.bitwise_and(edges, mask)
    return roi_edges


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return result


# Tạo Camera
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định (0)

while True:
    ret, frame = cap.read()  # Đọc frame từ camera
    if not ret:
        break

    processed_frame = process_frame(frame)  # Xử lý frame

    cv2.imshow('Processed Frame', processed_frame)  # Hiển thị frame đã xử lý

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

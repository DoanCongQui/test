import cv2 as cv
import numpy as np
from time import time

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv.addWeighted(initial_img, α, img, β, λ)

def detect_line_segments(cropped_edges):
    rho = 1
    angle = np.pi / 180
    min_threshold = 10
    line_segments = cv.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                   np.array([]), minLineLength=10, maxLineGap=15)
    return line_segments

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def average_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def display_lines(frame, lines, line_color=(0, 255, 255), line_width=20):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)
    cv.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    return cropped_edges

# Định nghĩa khoảng màu đen trong không gian màu HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([227, 100, 70])

# Kích thước kernel cho việc dilate
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))

# Sử dụng camera tích hợp trên laptop
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

print('Started')
print("Beginning Transmitting to channel: Happy_Robots")
start_time = time()

while True:
    try:
        # Đọc từng frame từ camera
        ret, frame = cap.read()

        if not ret:
            break

        # Áp dụng một số gaussian blur cho hình ảnh
        kernel_size = (3, 3)
        gauss_image = cv.GaussianBlur(frame, kernel_size, 0)

        # Chuyển sang không gian màu HSV
        hsv_image = cv.cvtColor(gauss_image, cv.COLOR_BGR2HSV)

        # Áp dụng ngưỡng màu để chỉ nhận các màu đen
        thres_1 = cv.inRange(hsv_image, lower_black, upper_black)

        # Dilate ảnh ngưỡng
        thresh = cv.dilate(thres_1, rect_kernel, iterations=1)

        # Áp dụng canny edge detection
        low_threshold = 200
        high_threshold = 400
        canny_edges = cv.Canny(gauss_image, low_threshold, high_threshold)

        # Lấy vùng quan tâm
        roi_image = region_of_interest(canny_edges)

        # Detect các đoạn thẳng trong ảnh
        line_segments = detect_line_segments(roi_image)

        # Kết hợp các đoạn thẳng thành đường lái
        lane_lines = average_slope_intercept(frame, line_segments)

        # Hiển thị ảnh với các đường lái
        line_image = display_lines(frame, lane_lines)

        # Hiển thị cả frame gốc và các kết quả xử lý
        cv.imshow('Frame', frame)
        cv.imshow('New Image', roi_image)
        cv.imshow('Line Image', line_image)

        # Đợi 30ms và kiểm tra nút thoát
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    except KeyboardInterrupt:
        break

# Giải phóng tài nguyên
cap.release()
cv.destroyAllWindows()

print('Stopped')

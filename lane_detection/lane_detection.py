import numpy as np
import cv2 as cv


# TODO: Merge nearby lines
# TODO: Only display lines that belong to right lane
# TODO: Display the lane more beautiful, e.g. area between lines


def calculate_slope(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line.squeeze()
    slope = np.arctan2(y2 - y1, x2 - x1)

    return np.degrees(slope)


def find_lines(frame: np.ndarray) -> np.ndarray:
    # Convert to grayscale and detect edges
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 100, 200)

    # Define region of interest and mask frame
    roi = np.array([[[0, 600], [1280, 600], [600, 300]]])
    mask = np.zeros_like(canny)
    cv.fillPoly(mask, roi, 255)
    masked_frame = cv.bitwise_and(canny, mask)

    # Detect lines (Probabilistic Hough Transform works better)
    lines = cv.HoughLinesP(masked_frame, 2, np.pi/180, 100, np.array([]),
                           minLineLength=50, maxLineGap=200)

    return lines


def show_lines(frame: np.ndarray, lines: np.ndarray) -> None:
    if lines is not None:
        for line in lines:
            if abs(calculate_slope(line)) > 20:
                x1, y1, x2, y2 = line.squeeze()
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255))


if __name__ == "__main__":
    cap = cv.VideoCapture("videos/test_simple.mp4")

    while cap.isOpened():
        _, frame = cap.read()
        lines = find_lines(frame)
        show_lines(frame, lines)
        cv.imshow("Dash Cam", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

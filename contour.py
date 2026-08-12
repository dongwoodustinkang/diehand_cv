"""A/B TIFF의 B 페이지에서 컨투어와 첫 접점을 찾는다."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


MIN_CONTOUR_AREA = 4000
MAX_CONTOUR_AREA = 10000
SAMPLE_COUNT = 15
POINT_RADIUS = 3

CONTOUR_COLOR = (0, 255, 0)
TOP_COLOR = (255, 0, 0)
BOTTOM_COLOR = (0, 0, 255)
LEFT_COLOR = (255, 0, 255)
RIGHT_COLOR = (0, 255, 255)


@dataclass
class ContourMeasurement:
    """컨투어 하나의 4면 첫 접점 정보."""

    contour: np.ndarray
    bounding_rect: Tuple[int, int, int, int]
    top_points: List[Tuple[int, int]]
    bottom_points: List[Tuple[int, int]]
    left_points: List[Tuple[int, int]]
    right_points: List[Tuple[int, int]]


@dataclass
class DetectionResult:
    """B 페이지 컨투어 분석 결과."""

    contours: List[np.ndarray] = field(default_factory=list)
    measurements: List[ContourMeasurement] = field(default_factory=list)


def load_tiff_pages(image_path):
    """TIFF의 A, B 페이지를 읽는다."""

    success, images = cv2.imreadmulti(str(image_path), flags=cv2.IMREAD_UNCHANGED)
    if not success or len(images) < 2:
        raise ValueError(f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}")

    return images[0], images[1]


def to_grayscale(image):
    """이미지를 그레이스케일로 변환한다."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_b_contours(image_b):
    """그레이 스케일, 이진화, 컨투어링 작업."""

    gray = to_grayscale(image_b)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
    filtered_contours = [
        contour for contour in contours
        if MIN_CONTOUR_AREA <= cv2.contourArea(contour) <= MAX_CONTOUR_AREA
    ]
    return DetectionResult(contours=filtered_contours)


def get_sample_positions(start, size):
    """컨투어 한 변을 15개 위치로 나눈다."""

    end = start + max(size - 1, 0)
    return np.linspace(start, end, SAMPLE_COUNT, dtype=int)


def find_first_contact_points(contour, image_shape):
    """컨투어의 상·하·좌·우 첫 접점을 각각 15개씩 찾는다."""

    image_height, image_width = image_shape
    x, y, width, height = cv2.boundingRect(contour)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    top_points = []
    bottom_points = []
    for point_x in get_sample_positions(x, width):
        y_positions = np.where(mask[:, point_x] == 255)[0]
        if len(y_positions) == 0:
            continue

        top_points.append((int(point_x), int(y_positions[0])))
        bottom_points.append((int(point_x), int(y_positions[-1])))

    left_points = []
    right_points = []
    for point_y in get_sample_positions(y, height):
        x_positions = np.where(mask[point_y, :] == 255)[0]
        if len(x_positions) == 0:
            continue

        left_points.append((int(x_positions[0]), int(point_y)))
        right_points.append((int(x_positions[-1]), int(point_y)))

    return ContourMeasurement(
        contour=contour,
        bounding_rect=(x, y, width, height),
        top_points=top_points,
        bottom_points=bottom_points,
        left_points=left_points,
        right_points=right_points,
    )


def draw_points(image, points, color):
    """점 목록을 이미지 위에 표시한다."""

    for point in points:
        cv2.circle(image, point, POINT_RADIUS, color, thickness=cv2.FILLED)


def draw_detection_result(image_b, result):
    """B 페이지에 컨투어와 4면 첫 접점을 표시한다."""

    result_image = cv2.cvtColor(to_grayscale(image_b), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, result.contours, -1, CONTOUR_COLOR, 1)

    for measurement in result.measurements:
        draw_points(result_image, measurement.top_points, TOP_COLOR)
        draw_points(result_image, measurement.bottom_points, BOTTOM_COLOR)
        draw_points(result_image, measurement.left_points, LEFT_COLOR)
        draw_points(result_image, measurement.right_points, RIGHT_COLOR)

    return result_image


def create_detection_visualization(image_path):
    """B 페이지 컨투어와 첫 접점 이미지를 만든다."""

    _, image_b = load_tiff_pages(Path(image_path))
    result = find_b_contours(image_b)
    gray = to_grayscale(image_b)

    for contour in result.contours:
        measurement = find_first_contact_points(contour, gray.shape)
        result.measurements.append(measurement)

    result_image = draw_detection_result(image_b, result)
    return result_image, result

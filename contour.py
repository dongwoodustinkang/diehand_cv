from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 검출할 컨투어의 최소/최대 크기
MIN_CONTOUR_AREA = 4000 
MAX_CONTOUR_AREA = 10000

# 샘플링 개수 설정
TOP_BOTTOM_SAMPLE_COUNT = 15
LEFT_RIGHT_SAMPLE_COUNT = 5

# 점(원)의 크기 조절
POINT_RADIUS = 2
EXTREME_POINT_RADIUS = 4
SECONDARY_POINT_RADIUS = 4
LINE_THICKNESS = 1        # 확장 직선 두께

CONTOUR_COLOR = (0, 255, 0)
TOP_COLOR = (255, 0, 0)
BOTTOM_COLOR = (0, 0, 255)
LEFT_COLOR = (255, 0, 255)
RIGHT_COLOR = (0, 255, 255)

# 시각화 색상 설정
EXTREME_POINT_COLOR = (0, 255, 255)         # 절대 극단점 (노란색)
SECONDARY_POINT_COLOR = (0, 165, 255)       # 보조 극단점 (주황색)


@dataclass
class ContourMeasurement:
    """컨투어 하나의 4면 첫 접점, 절대 극단점 및 보조 극단점 정보."""

    bounding_rect: Tuple[int, int, int, int]
    top_points: List[Tuple[int, int]]
    bottom_points: List[Tuple[int, int]]
    left_points: List[Tuple[int, int]]
    right_points: List[Tuple[int, int]]

    # 1. 절대 극단점 (4개)
    top_extreme: Tuple[int, int] = None
    bottom_extreme: Tuple[int, int] = None
    left_extreme: Tuple[int, int] = None
    right_extreme: Tuple[int, int] = None

    # 2. 보조 극단점 (3px 이내, 50px 이상 떨어진 점, 4개)
    top_secondary: Tuple[int, int] = None
    bottom_secondary: Tuple[int, int] = None
    left_secondary: Tuple[int, int] = None
    right_secondary: Tuple[int, int] = None


@dataclass
class DetectionResult:
    """B 페이지 컨투어 분석 결과."""

    contours: List[np.ndarray] = field(default_factory=list)
    measurements: List[ContourMeasurement] = field(default_factory=list)
    crop_preview: Optional[np.ndarray] = None


def to_grayscale(image):
    """이미지를 그레이스케일로 변환한다."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_b_contours(image_b):
    """그레이 스케일, 이진화, 컨투어링 작업."""
    gray = to_grayscale(image_b) # 그레이스케일
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU,) # 이진화
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,) # 컨투어링 작업

    # 범위내 컨투어 검출
    filtered_contours = [
        con for con in contours
        if MIN_CONTOUR_AREA <= cv2.contourArea(con) <= MAX_CONTOUR_AREA
    ]

    return DetectionResult(contours=filtered_contours)


def find_first_contact_points(contour, image_shape):
    """
    ## 컨투어의 상하좌우 접점 찾기
    1. 컨투어를 그린다.
    2. 상하단 15개, 좌우측 5개로 포인트를 샘플링한다. (중복점 제거 안 함)
    3. 각 포인트에 대해 가장 가까운 절대 극단점(4개)을 찾는다.  
    
    """

    image_height, image_width = image_shape
    x, y, width, height = cv2.boundingRect(contour) # 컨투어의 x,y, width, height를 구함

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED) # 그대로 컨투어 그리기

    # 1. 샘플링 로직 (상·하단 15개, 좌·우측 5개)
    top_points = []
    bottom_points = []
    x_positions = np.linspace(x, x + width - 1, TOP_BOTTOM_SAMPLE_COUNT, dtype=int)
    for point_x in x_positions:
        y_positions = np.where(mask[:, point_x] == 255)[0]
        if len(y_positions) == 0:
            continue

        top_points.append((int(point_x), int(y_positions[0])))
        bottom_points.append((int(point_x), int(y_positions[-1])))

    left_points = []
    right_points = []
    y_positions = np.linspace(y, y + height - 1, LEFT_RIGHT_SAMPLE_COUNT, dtype=int)
    for point_y in y_positions:
        x_positions = np.where(mask[point_y, :] == 255)[0]
        if len(x_positions) == 0:
            continue

        left_points.append((int(x_positions[0]), int(point_y)))
        right_points.append((int(x_positions[-1]), int(point_y)))

    # 2. 절대 극단점 추출 로직 (기존 4개)
    pts = contour[:, 0, :]  # (N, 2) 형태의 좌표 배열

    top_idx = pts[:, 1].argmin()
    bottom_idx = pts[:, 1].argmax()
    left_idx = pts[:, 0].argmin()
    right_idx = pts[:, 0].argmax()

    top_extreme = tuple(pts[top_idx])
    bottom_extreme = tuple(pts[bottom_idx])
    left_extreme = tuple(pts[left_idx])
    right_extreme = tuple(pts[right_idx])

    # 3. 보조 극단점 추출 로직 (정도 차이 3px 이내, 50px 이상 떨어진 점)
    def find_secondary_point(candidates_mask, extreme_pt, min_distance=50):
        candidate_indices = np.where(candidates_mask)[0]
        valid_pts = []
        for idx in candidate_indices:
            pt = pts[idx]
            dist = np.linalg.norm(pt - np.array(extreme_pt))
            if dist >= min_distance:
                valid_pts.append((dist, tuple(pt)))

        if not valid_pts:
            if len(candidate_indices) > 0:
                dists = [np.linalg.norm(pts[i] - np.array(extreme_pt)) for i in candidate_indices]
                max_i = np.argmax(dists)
                return tuple(pts[candidate_indices[max_i]])
            return None

        valid_pts.sort(key=lambda x: x[0], reverse=True)
        return valid_pts[0][1]

    top_secondary = find_secondary_point(pts[:, 1] <= pts[top_idx, 1] + 3, top_extreme, min_distance=50)
    bottom_secondary = find_secondary_point(pts[:, 1] >= pts[bottom_idx, 1] - 3, bottom_extreme, min_distance=50)
    left_secondary = find_secondary_point(pts[:, 0] <= pts[left_idx, 0] + 3, left_extreme, min_distance=50)
    right_secondary = find_secondary_point(pts[:, 0] >= pts[right_idx, 0] - 3, right_extreme, min_distance=50)

    return ContourMeasurement(
        bounding_rect=(x, y, width, height),
        top_points=top_points,
        bottom_points=bottom_points,
        left_points=left_points,
        right_points=right_points,
        top_extreme=top_extreme,
        bottom_extreme=bottom_extreme,
        left_extreme=left_extreme,
        right_extreme=right_extreme,
        top_secondary=top_secondary,
        bottom_secondary=bottom_secondary,
        left_secondary=left_secondary,
        right_secondary=right_secondary,
    )


def draw_extended_line(img, pt1, pt2, color, thickness=1):
    """두 점을 지나는 직선을 이미지 경계(끝)까지 연장하여 그린다."""
    if pt1 is None or pt2 is None:
        return

    h, w = img.shape[:2]
    x1, y1 = pt1
    x2, y2 = pt2

    if x1 == x2 and y1 == y2:
        return

    dx = x2 - x1
    dy = y2 - y1

    t_values = []
    if dx != 0:
        t_values.append((0 - x1) / dx)
        t_values.append(((w - 1) - x1) / dx)
    if dy != 0:
        t_values.append((0 - y1) / dy)
        t_values.append(((h - 1) - y1) / dy)

    if not t_values:
        return

    valid_points = []
    for t in t_values:
        x = x1 + t * dx
        y = y1 + t * dy
        if -0.5 <= x <= w - 0.5 and -0.5 <= y <= h - 0.5:
            cx = int(np.clip(round(x), 0, w - 1))
            cy = int(np.clip(round(y), 0, h - 1))
            valid_points.append((cx, cy))

    unique_pts = []
    for p in valid_points:
        if p not in unique_pts:
            unique_pts.append(p)

    if len(unique_pts) >= 2:
        max_dist = 0
        p_a, p_b = unique_pts[0], unique_pts[1]
        for i in range(len(unique_pts)):
            for j in range(i + 1, len(unique_pts)):
                dist = (unique_pts[i][0] - unique_pts[j][0])**2 + (unique_pts[i][1] - unique_pts[j][1])**2
                if dist > max_dist:
                    max_dist = dist
                    p_a = unique_pts[i]
                    p_b = unique_pts[j]
        cv2.line(img, p_a, p_b, color, thickness)


def find_line_intersection(line1_start, line1_end, line2_start, line2_end):
    """두 직선의 교차점을 반환한다. 평행한 경우 None을 반환한다."""

    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None

    determinant1 = x1 * y2 - y1 * x2
    determinant2 = x3 * y4 - y3 * x4
    point_x = (determinant1 * (x3 - x4) - (x1 - x2) * determinant2) / denominator
    point_y = (determinant1 * (y3 - y4) - (y1 - y2) * determinant2) / denominator
    return point_x, point_y


def crop_between_extended_lines(image_b, measurement):
    """상·하·좌·우 연장선의 내부 영역을 직사각형으로 펼쳐 Crop한다."""

    lines = [
        (measurement.top_extreme, measurement.top_secondary),
        (measurement.bottom_extreme, measurement.bottom_secondary),
        (measurement.left_extreme, measurement.left_secondary),
        (measurement.right_extreme, measurement.right_secondary),
    ]
    if any(start is None or end is None for start, end in lines):
        return None

    top_line, bottom_line, left_line, right_line = lines
    corners = [
        find_line_intersection(*top_line, *left_line),
        find_line_intersection(*top_line, *right_line),
        find_line_intersection(*bottom_line, *right_line),
        find_line_intersection(*bottom_line, *left_line),
    ]
    if any(corner is None for corner in corners):
        return None

    source_points = np.float32(corners)
    top_width = np.linalg.norm(source_points[1] - source_points[0])
    bottom_width = np.linalg.norm(source_points[2] - source_points[3])
    left_height = np.linalg.norm(source_points[3] - source_points[0])
    right_height = np.linalg.norm(source_points[2] - source_points[1])
    crop_width = max(1, round(max(top_width, bottom_width)))
    crop_height = max(1, round(max(left_height, right_height)))

    target_points = np.float32(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(source_points, target_points)
    source_image = cv2.cvtColor(to_grayscale(image_b), cv2.COLOR_GRAY2BGR)
    return cv2.warpPerspective(source_image, transform, (crop_width, crop_height))


def create_crop_preview(image_b, measurements):
    """컨투어별 Crop 이미지를 세로로 합쳐 사이드바 미리보기로 만든다."""

    crops = []
    for measurement in measurements:
        cropped_image = crop_between_extended_lines(image_b, measurement)
        if cropped_image is not None and cropped_image.size > 0:
            crops.append(cropped_image)

    if not crops:
        return None
    if len(crops) == 1:
        return crops[0]

    gap = 8
    preview_width = max(crop.shape[1] for crop in crops)
    preview_height = sum(crop.shape[0] for crop in crops) + gap * (len(crops) - 1)
    preview = np.full((preview_height, preview_width, 3), 249, dtype=np.uint8)

    offset_y = 0
    for crop in crops:
        offset_x = (preview_width - crop.shape[1]) // 2
        preview[offset_y : offset_y + crop.shape[0], offset_x : offset_x + crop.shape[1]] = crop
        offset_y += crop.shape[0] + gap

    return preview


def create_detection_visualization(image_path):
    """B 페이지 컨투어, 샘플 접점, 극단점, 보조점 및 연장된 직선 이미지를 만든다."""

    success, images = cv2.imreadmulti(str(image_path), flags=cv2.IMREAD_UNCHANGED)
    if not success or len(images) < 2:
        raise ValueError(f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}")

    image_b = images[1]
    gray = to_grayscale(image_b) # 이미지 파일 그레이 스케일 
    result = find_b_contours(image_b) # 컨투어링 작업

    for contour in result.contours:
        # 컨투어에 가장 먼저 닿는 접점 찾기
        measurement = find_first_contact_points(contour, gray.shape)
        result.measurements.append(measurement)

    result.crop_preview = create_crop_preview(image_b, result.measurements) # 사이드바 프리뷰
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 그레이 스케일를 컬로로 변환하여 컨투어 색상이 보이게
    cv2.drawContours(result_image, result.contours, -1, CONTOUR_COLOR, 1) # 컨투어 그리기 

    for measurement in result.measurements:
        # 1. 상·하·좌·우 두 점 연결 직선 그리기 (이미지 끝까지 연장)
        draw_extended_line(result_image, measurement.top_extreme, measurement.top_secondary, TOP_COLOR, LINE_THICKNESS)
        draw_extended_line(result_image, measurement.bottom_extreme, measurement.bottom_secondary, BOTTOM_COLOR, LINE_THICKNESS)
        draw_extended_line(result_image, measurement.left_extreme, measurement.left_secondary, LEFT_COLOR, LINE_THICKNESS)
        draw_extended_line(result_image, measurement.right_extreme, measurement.right_secondary, RIGHT_COLOR, LINE_THICKNESS)

        # 2. 샘플 점들 그리기
        point_groups = [
            (measurement.top_points, TOP_COLOR),
            (measurement.bottom_points, BOTTOM_COLOR),
            (measurement.left_points, LEFT_COLOR),
            (measurement.right_points, RIGHT_COLOR),
        ]
        for points, color in point_groups:
            for point in points:
                cv2.circle(result_image, point, POINT_RADIUS, color, thickness=cv2.FILLED)

        # 3. 절대 극단점 그리기 (노란색)
        extreme_points = [
            measurement.top_extreme,
            measurement.bottom_extreme,
            measurement.left_extreme,
            measurement.right_extreme,
        ]
        for ex_point in extreme_points:
            if ex_point is not None:
                cv2.circle(result_image, ex_point, EXTREME_POINT_RADIUS, EXTREME_POINT_COLOR, thickness=cv2.FILLED)
                cv2.circle(result_image, ex_point, EXTREME_POINT_RADIUS + 1, (0, 0, 0), thickness=1)

        # 4. 보조 극단점 그리기 (주황색)
        secondary_points = [
            measurement.top_secondary,
            measurement.bottom_secondary,
            measurement.left_secondary,
            measurement.right_secondary,
        ]
        for sec_point in secondary_points:
            if sec_point is not None:
                cv2.circle(result_image, sec_point, SECONDARY_POINT_RADIUS, SECONDARY_POINT_COLOR, thickness=cv2.FILLED)
                cv2.circle(result_image, sec_point, SECONDARY_POINT_RADIUS + 1, (0, 0, 0), thickness=1)

    return result_image, result

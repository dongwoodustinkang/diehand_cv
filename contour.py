from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 검출할 컨투어의 최소/최대 크기
MIN_CONTOUR_AREA = 3500
MAX_CONTOUR_AREA = 10000

# 샘플링 개수 설정
TOP_BOTTOM_SAMPLE_COUNT = 15
LEFT_RIGHT_SAMPLE_COUNT = 5

# 점(원)의 크기 조절
POINT_RADIUS = 1
EXTREME_POINT_RADIUS = 3
SECONDARY_POINT_RADIUS = 3
LINE_THICKNESS = 1        # 확장 직선 두께
EXTREME_SENCODARY_DIFF = 2
MIN_DISTANCE = 30
APPROX_POLYGON_EPSILON_RATIO = 0.02
PREVIEW_MASK_MODE = "approx_polygon"
# "contour", "approx_polygon", 또는 "line_quadrilateral"
LINE_QUADRILATERAL_METHOD = "robust_contacts"
# "robust_contacts" 또는 이전 방식인 "extreme_pairs"

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
    analysis_preview: Optional[np.ndarray] = None
    source_preview: Optional[np.ndarray] = None


def to_grayscale(image):
    """이미지를 그레이스케일로 변환한다."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image):
    """미리보기에 사용할 이미지를 BGR 3채널로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def to_bgra(image):
    """투명 Preview에 사용할 이미지를 BGRA 4채널로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image.copy()


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
    def find_secondary_point(candidates_mask, extreme_pt, min_distance=MIN_DISTANCE):
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


    # PIXEL 정도 차이
    top_secondary = find_secondary_point(pts[:, 1] <= pts[top_idx, 1] + EXTREME_SENCODARY_DIFF, top_extreme, min_distance=MIN_DISTANCE)
    bottom_secondary = find_secondary_point(pts[:, 1] >= pts[bottom_idx, 1] - EXTREME_SENCODARY_DIFF, bottom_extreme, min_distance=MIN_DISTANCE)
    left_secondary = find_secondary_point(pts[:, 0] <= pts[left_idx, 0] + EXTREME_SENCODARY_DIFF, left_extreme, min_distance=MIN_DISTANCE)
    right_secondary = find_secondary_point(pts[:, 0] >= pts[right_idx, 0] - EXTREME_SENCODARY_DIFF, right_extreme, min_distance=MIN_DISTANCE)

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


def stack_preview_tiles(tiles):
    """여러 BGRA Preview 타일을 투명한 세로 Preview로 합친다."""

    if not tiles:
        return None
    if len(tiles) == 1:
        return tiles[0]

    gap = 8
    preview_width = max(tile.shape[1] for tile in tiles)
    preview_height = sum(tile.shape[0] for tile in tiles) + gap * (len(tiles) - 1)
    preview = np.zeros((preview_height, preview_width, 4), dtype=np.uint8)

    offset_y = 0
    for tile in tiles:
        tile_height, tile_width = tile.shape[:2]
        offset_x = (preview_width - tile_width) // 2
        preview[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width] = tile
        offset_y += tile_height + gap
    return preview


def approximate_contour_polygon(contour):
    """컨투어를 둘레 길이 비율 기반의 다각형으로 근사한다."""

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    polygon = cv2.approxPolyDP(
        contour, APPROX_POLYGON_EPSILON_RATIO * perimeter, True
    )
    return polygon if len(polygon) >= 3 else None


def get_line_intersection(first_start, first_end, second_start, second_end):
    """두 기준선의 교점을 반환하고, 평행하면 ``None``을 반환한다."""

    first_start = np.asarray(first_start, dtype=float)
    first_direction = np.asarray(first_end, dtype=float) - first_start
    second_start = np.asarray(second_start, dtype=float)
    second_direction = np.asarray(second_end, dtype=float) - second_start
    denominator = np.cross(first_direction, second_direction)
    if np.isclose(denominator, 0):
        return None

    distance = np.cross(second_start - first_start, second_direction) / denominator
    return tuple(np.rint(first_start + distance * first_direction).astype(int))


def fit_line_from_contact_points(points, orientation):
    """방향별 중앙값 이상점을 제외한 접점으로 기준선을 피팅한다."""

    if len(points) < 2:
        return None

    samples = np.asarray(points, dtype=np.float32)
    coordinate_index = 1 if orientation == "horizontal" else 0
    coordinate_values = samples[:, coordinate_index]
    median = np.median(coordinate_values)
    median_absolute_deviation = np.median(np.abs(coordinate_values - median))
    tolerance = max(2.0, 3.0 * median_absolute_deviation)
    inliers = samples[np.abs(coordinate_values - median) <= tolerance]
    if len(inliers) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        inliers.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
    )
    direction = np.array([float(vx), float(vy)])
    if np.linalg.norm(direction) == 0:
        return None
    origin = np.array([float(x0), float(y0)])
    return tuple(origin - direction), tuple(origin + direction)


def get_extreme_pair_lines(measurement):
    """기존 극점·보조점 쌍으로 만든 네 기준선을 반환한다."""

    return (
        (measurement.top_extreme, measurement.top_secondary),
        (measurement.bottom_extreme, measurement.bottom_secondary),
        (measurement.left_extreme, measurement.left_secondary),
        (measurement.right_extreme, measurement.right_secondary),
    )


def get_robust_contact_lines(measurement):
    """상·하·좌·우 접점 집합으로 피팅한 네 기준선을 반환한다."""

    return (
        fit_line_from_contact_points(measurement.top_points, "horizontal"),
        fit_line_from_contact_points(measurement.bottom_points, "horizontal"),
        fit_line_from_contact_points(measurement.left_points, "vertical"),
        fit_line_from_contact_points(measurement.right_points, "vertical"),
    )


def is_valid_line_quadrilateral(corners, measurement):
    """교점 사각형이 컨투어 근방의 볼록한 도형인지 확인한다."""

    if any(corner is None for corner in corners):
        return False

    polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 0:
        return False

    x, y, width, height = measurement.bounding_rect
    margin_x = max(12, round(width * 0.2))
    margin_y = max(12, round(height * 0.2))
    points = polygon[:, 0, :]
    return (
        points[:, 0].min() >= x - margin_x
        and points[:, 0].max() <= x + width - 1 + margin_x
        and points[:, 1].min() >= y - margin_y
        and points[:, 1].max() <= y + height - 1 + margin_y
    )


def get_line_quadrilateral(measurement):
    """상·하·좌·우 기준선의 교점 네 개를 시계 방향으로 반환한다."""

    if LINE_QUADRILATERAL_METHOD == "robust_contacts":
        lines = get_robust_contact_lines(measurement)
    elif LINE_QUADRILATERAL_METHOD == "extreme_pairs":
        lines = get_extreme_pair_lines(measurement)
    else:
        raise ValueError(
            f"지원하지 않는 기준선 사각형 방식: {LINE_QUADRILATERAL_METHOD}"
        )

    if any(line is None or any(point is None for point in line) for line in lines):
        return None
    top, bottom, left, right = lines

    corners = (
        get_line_intersection(*top, *left),
        get_line_intersection(*top, *right),
        get_line_intersection(*bottom, *right),
        get_line_intersection(*bottom, *left),
    )
    if any(corner is None for corner in corners):
        return None
    if (
        LINE_QUADRILATERAL_METHOD == "robust_contacts"
        and not is_valid_line_quadrilateral(corners, measurement)
    ):
        return None
    return np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)


def get_preview_mask_shape(contour, measurement):
    """현재 설정에 맞는 Preview 마스크 도형을 반환한다."""
    if PREVIEW_MASK_MODE == "contour":
        return contour
    if PREVIEW_MASK_MODE == "approx_polygon":
        return approximate_contour_polygon(contour)
    if PREVIEW_MASK_MODE == "line_quadrilateral":
        quadrilateral = get_line_quadrilateral(measurement)
        return quadrilateral if quadrilateral is not None else contour
    raise ValueError(f"지원하지 않는 Preview 마스크 방식: {PREVIEW_MASK_MODE}")


def get_polygon_crop_bounds(polygon, image_shape):
    """근사 다각형을 포함하는 최소 Crop 범위를 반환한다."""

    image_height, image_width = image_shape
    points = polygon[:, 0, :]
    left = max(0, int(points[:, 0].min()))
    top = max(0, int(points[:, 1].min()))
    right = min(image_width, int(points[:, 0].max()) + 1)
    bottom = min(image_height, int(points[:, 1].max()) + 1)
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def create_polygon_preview(image, contours, measurements):
    """선택한 마스크 도형 내부의 원본 픽셀만 남긴 투명 Preview를 만든다."""

    tiles = []
    for contour, measurement in zip(contours, measurements):
        polygon = get_preview_mask_shape(contour, measurement)
        if polygon is None:
            continue

        bounds = get_polygon_crop_bounds(polygon, image.shape[:2])
        if bounds is None:
            continue

        left, top, right, bottom = bounds
        tile = to_bgra(image)[top:bottom, left:right].copy()
        mask = np.zeros(tile.shape[:2], dtype=np.uint8)
        local_polygon = polygon.copy()
        local_polygon[:, 0, 0] -= left
        local_polygon[:, 0, 1] -= top
        cv2.fillPoly(mask, [local_polygon], 255, lineType=cv2.LINE_AA)
        tile[:, :, 3] = cv2.bitwise_and(tile[:, :, 3], mask)
        tiles.append(tile)
    return stack_preview_tiles(tiles)


def create_detection_visualization(image_path):
    """B 분석선과 동일 좌표의 A 페이지 Crop 미리보기를 만든다."""

    success, images = cv2.imreadmulti(str(image_path), flags=cv2.IMREAD_UNCHANGED)
    if not success or len(images) < 2:
        raise ValueError(f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}")

    image_a, image_b = images[:2]
    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(f"A/B 페이지 크기가 일치하지 않습니다: {image_path}")

    gray = to_grayscale(image_b) # 이미지 파일 그레이 스케일 
    result = find_b_contours(image_b) # 컨투어링 작업

    for contour in result.contours:
        # 컨투어에 가장 먼저 닿는 접점 찾기
        measurement = find_first_contact_points(contour, gray.shape)
        result.measurements.append(measurement)

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

    result.analysis_preview = create_polygon_preview(
        image_b, result.contours, result.measurements
    )
    result.source_preview = create_polygon_preview(
        image_a, result.contours, result.measurements
    )
    return result_image, result

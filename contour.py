from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# 검출할 컨투어의 최소/최대 크기
MIN_CONTOUR_AREA = 3500
MAX_CONTOUR_AREA = 10000

# 기존 균등 샘플 수 (CONTACT_POINT_MODE = "sampled"일 때만 사용)
TOP_BOTTOM_SAMPLE_COUNT = 15
LEFT_RIGHT_SAMPLE_COUNT = 5
CONTACT_POINT_MODE = "full"  # "full"이면 전체 접점 분포, "sampled"면 기존 15/5 샘플
TOP_DENSE_BAND_TOLERANCE = 1

# 점(원)의 크기 조절
REFERENCE_POINT_RADIUS = 2
REFERENCE_POINT_MIN_DISTANCE = 30  # 실제 보조 접점과의 최소 거리(px)
EXTREME_SENCODARY_DIFF = 2
MIN_DISTANCE = 30
APPROX_POLYGON_EPSILON_RATIO = 0.02
ANALYSIS_PREVIEW_MASK_MODE = "contour"
SOURCE_PREVIEW_MASK_MODE = "line_quadrilateral"
SOURCE_PREVIEW_OUTER_MARGIN = 1
SOURCE_PREVIEW_TOP_SCAN_HEIGHT = 3
SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD = 40
SOURCE_PREVIEW_EDGE_CHANGE_WINDOW = 3
SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN = 3
SOURCE_PREVIEW_INVALID_Y_DIFFERENCE = 10
# "contour", "approx_polygon", 또는 "line_quadrilateral"
LINE_QUADRILATERAL_METHOD = "primary_axis"
# "primary_axis", "robust_contacts" 또는 이전 방식인 "extreme_pairs"

CONTOUR_COLOR = (247, 85, 168)  # 보라색 (BGR)
TOP_COLOR = (248, 189, 56)  # 하늘색 (BGR)
BOTTOM_COLOR = (0, 0, 255)
LEFT_COLOR = (0, 255, 0)
RIGHT_COLOR = (0, 255, 255)
PILLAR_DOWNWARD_POINT_COLOR = (0, 0, 255)
PILLAR_POINT_RADIUS = 4

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

    # 2. 보조 극단점 (2px 이내, 30px 이상 떨어진 점, 4개)
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
    source_visualization: Optional[np.ndarray] = None


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
    2. 전체 또는 균등 샘플 위치에서 상하좌우 첫 접점을 수집한다.
    3. 각 포인트에 대해 가장 가까운 절대 극단점(4개)을 찾는다.  
    
    """

    image_height, image_width = image_shape
    x, y, width, height = cv2.boundingRect(contour) # 컨투어의 x,y, width, height를 구함

    # 채워진 내부 면적이 아니라 실제 외곽선만 사용한다. 히스토그램과 기준선은
    # 내부 픽셀이 아닌 각 방향에서 외곽선을 처음 만나는 지점만 사용한다.
    boundary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(
        boundary_mask,
        [contour],
        -1,
        255,
        thickness=1,
        lineType=cv2.LINE_8,
    )

    # 1. 접점 수집: 전체 분포 또는 기존 균등 샘플
    top_points = []
    bottom_points = []
    if CONTACT_POINT_MODE == "full":
        x_positions = range(x, x + width)
        y_positions = range(y, y + height)
    elif CONTACT_POINT_MODE == "sampled":
        x_positions = np.linspace(
            x, x + width - 1, TOP_BOTTOM_SAMPLE_COUNT, dtype=int
        )
        y_positions = np.linspace(
            y, y + height - 1, LEFT_RIGHT_SAMPLE_COUNT, dtype=int
        )
    else:
        raise ValueError(f"지원하지 않는 접점 수집 방식: {CONTACT_POINT_MODE}")


    # 상면 하면 접점 중에
    for point_x in x_positions:
        contact_y_positions = np.where(boundary_mask[:, point_x] == 255)[0] # 고정된 x 좌표에 색상이 255인 y좌표의 배열
        if len(contact_y_positions) == 0: # 만약 y좌표가 존재하지 않으면 pass
            continue

        top_points.append((int(point_x), int(contact_y_positions[0]))) # 가장 상위 좌표
        bottom_points.append((int(point_x), int(contact_y_positions[-1]))) # 가장 하위 좌표

    left_points = []
    right_points = []
    for point_y in y_positions:
        contact_x_positions = np.where(boundary_mask[point_y, :] == 255)[0]
        if len(contact_x_positions) == 0:
            continue

        left_points.append((int(contact_x_positions[0]), int(point_y)))
        right_points.append((int(contact_x_positions[-1]), int(point_y)))

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


def get_densest_band_points(points, coordinate_index, tolerance):
    """최빈 좌표 주변의 가장 밀집한 점 띠를 반환한다."""

    samples = np.asarray(points, dtype=np.float32)
    if len(samples) < 2:
        return samples

    coordinates = np.rint(samples[:, coordinate_index]).astype(int)
    values, counts = np.unique(coordinates, return_counts=True)
    most_frequent_values = values[counts == counts.max()]
    median = np.median(coordinates)
    center = most_frequent_values[
        np.argmin(np.abs(most_frequent_values - median))
    ]
    return samples[np.abs(coordinates - center) <= tolerance]


def get_primary_contact_reference_point(points, coordinate_index, use_minimum):
    """첫 접점 중 가장 먼저 닿는 최솟값 또는 최댓값의 실제 대표점을 고른다."""

    if not points:
        return None

    coordinates = [point[coordinate_index] for point in points]
    coordinate = min(coordinates) if use_minimum else max(coordinates)
    candidates = [
        point for point in points if point[coordinate_index] == coordinate
    ]
    other_coordinate_index = 1 - coordinate_index
    median = np.median([point[other_coordinate_index] for point in candidates])
    return min(
        candidates,
        key=lambda point: (
            abs(point[other_coordinate_index] - median),
            point[other_coordinate_index],
        ),
    )


def draw_axis_aligned_reference_line(img, point, coordinate_index, color, thickness=1):
    """상·하는 수평선, 좌·우는 수직선을 대표점에서 이미지 끝까지 그린다."""

    if point is None:
        return

    image_height, image_width = img.shape[:2]
    x, y = point
    if coordinate_index == 1:
        cv2.line(img, (0, y), (image_width - 1, y), color, thickness)
    else:
        cv2.line(img, (x, 0), (x, image_height - 1), color, thickness)


def get_spaced_reference_point(points, reference_point, coordinate_index, min_distance):
    """같은 기준 좌표에서 충분히 떨어진 실제 첫 접점 하나를 반환한다."""

    if reference_point is None:
        return None

    other_coordinate_index = 1 - coordinate_index
    candidates = [
        point
        for point in points
        if point[coordinate_index] == reference_point[coordinate_index]
        and abs(point[other_coordinate_index] - reference_point[other_coordinate_index])
        >= min_distance
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda point: abs(
            point[other_coordinate_index] - reference_point[other_coordinate_index]
        ),
    )


def fit_line_from_contact_points(points, orientation, use_densest_band=False):
    """방향별 중앙값 이상점을 제외한 접점으로 기준선을 피팅한다."""

    if len(points) < 2:
        return None

    samples = np.asarray(points, dtype=np.float32)
    coordinate_index = 1 if orientation == "horizontal" else 0
    if use_densest_band:
        samples = get_densest_band_points(
            samples, coordinate_index, TOP_DENSE_BAND_TOLERANCE
        )
        if len(samples) < 2:
            return None

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
        fit_line_from_contact_points(
            measurement.top_points, "horizontal", use_densest_band=True
        ),
        fit_line_from_contact_points(measurement.bottom_points, "horizontal"),
        fit_line_from_contact_points(measurement.left_points, "vertical"),
        fit_line_from_contact_points(measurement.right_points, "vertical"),
    )


def get_primary_axis_quadrilateral(measurement):
    """B 이미지의 대표 수평·수직 기준선 교점으로 만든 축 정렬 사각형을 반환한다."""

    top_point = get_primary_contact_reference_point(
        measurement.top_points, 1, use_minimum=True
    )
    bottom_point = get_primary_contact_reference_point(
        measurement.bottom_points, 1, use_minimum=False
    )
    left_point = get_primary_contact_reference_point(
        measurement.left_points, 0, use_minimum=True
    )
    right_point = get_primary_contact_reference_point(
        measurement.right_points, 0, use_minimum=False
    )
    if any(point is None for point in (top_point, bottom_point, left_point, right_point)):
        return None

    top_y = top_point[1]
    bottom_y = bottom_point[1]
    left_x = left_point[0]
    right_x = right_point[0]
    if top_y >= bottom_y or left_x >= right_x:
        return None

    return np.asarray(
        (
            (left_x, top_y),
            (right_x, top_y),
            (right_x, bottom_y),
            (left_x, bottom_y),
        ),
        dtype=np.int32,
    ).reshape(-1, 1, 2)


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

    if LINE_QUADRILATERAL_METHOD == "primary_axis":
        return get_primary_axis_quadrilateral(measurement)
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


def expand_quadrilateral_by_side(polygon, margin):
    """사각형의 상·하·좌·우 변을 좌표축 방향으로 바깥쪽 확장한다."""

    if margin <= 0:
        return polygon

    points = polygon[:, 0, :].astype(float)
    if len(points) != 4:
        return polygon

    top_left, top_right, bottom_right, bottom_left = points
    top = (top_left + (0, -margin), top_right + (0, -margin))
    bottom = (bottom_right + (0, margin), bottom_left + (0, margin))
    left = (bottom_left + (-margin, 0), top_left + (-margin, 0))
    right = (top_right + (margin, 0), bottom_right + (margin, 0))

    corners = (
        get_line_intersection(*top, *left),
        get_line_intersection(*top, *right),
        get_line_intersection(*bottom, *right),
        get_line_intersection(*bottom, *left),
    )
    if any(corner is None for corner in corners):
        return polygon
    return np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)


def get_preview_mask_shape(contour, measurement, mask_mode):
    """지정한 방식에 맞는 Preview 마스크 도형을 반환한다."""
    if mask_mode == "contour":
        return contour
    if mask_mode == "approx_polygon":
        return approximate_contour_polygon(contour)
    if mask_mode == "line_quadrilateral":
        quadrilateral = get_line_quadrilateral(measurement)
        if quadrilateral is None:
            return contour
        return expand_quadrilateral_by_side(
            quadrilateral, SOURCE_PREVIEW_OUTER_MARGIN
        )
    raise ValueError(f"지원하지 않는 Preview 마스크 방식: {mask_mode}")


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


def find_top_pillar_reference_points(image):
    """원본 A 페이지 상단의 큰 밝기 변화 두 지점을 찾는다."""

    grayscale = to_grayscale(image)
    image_width = image.shape[1]
    image_height = image.shape[0]
    if image_width < 2 or image_height < SOURCE_PREVIEW_TOP_SCAN_HEIGHT:
        return None

    # 상단 몇 줄의 중앙값으로 노이즈를 줄인 뒤, x 방향 밝기 변화가 큰 첫·마지막
    # 지점을 각각 좌·우 기둥 경계로 사용한다.
    top_profile = np.median(
        grayscale[:SOURCE_PREVIEW_TOP_SCAN_HEIGHT, :], axis=0
    )
    # 인접 픽셀만 비교하면 완만한 경계가 누락될 수 있어, 좌·우 3 px 간격의
    # 밝기 차이를 사용한다. 좌/우 절반에서 변화량이 가장 큰 점이 기둥 경계다.
    edge_window = SOURCE_PREVIEW_EDGE_CHANGE_WINDOW
    if image_width <= edge_window * 2:
        return None
    horizontal_changes = np.zeros(image_width, dtype=np.float32)
    horizontal_changes[edge_window:-edge_window] = np.abs(
        top_profile[edge_window * 2 :] - top_profile[: -edge_window * 2]
    )
    midpoint_x = image_width // 2
    left_x = int(np.argmax(horizontal_changes[:midpoint_x]))
    right_x = int(midpoint_x + np.argmax(horizontal_changes[midpoint_x:]))
    if (
        horizontal_changes[left_x] < SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD
        or horizontal_changes[right_x] < SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD
    ):
        return None
    if left_x >= right_x:
        return None

    point_y = PILLAR_POINT_RADIUS
    return (
        (max(0, left_x - 5), point_y),
        (min(image_width - 1, right_x + 5), point_y),
    )


def find_downward_color_change_points(image, reference_points):
    """두 기준점의 y 차이가 허용 범위인 아래 방향 변화점 쌍을 찾는다."""

    if reference_points is None:
        return ()

    grayscale = to_grayscale(image)
    candidate_points = []
    for point_x, point_y in reference_points:
        vertical_profile = grayscale[point_y:, point_x].astype(np.int16)
        reference_brightness = vertical_profile[0]
        color_changed = np.abs(
            vertical_profile - reference_brightness
        ) >= SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD
        persistent_change = np.convolve(
            color_changed.astype(np.int16),
            np.ones(SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN, dtype=np.int16),
            mode="valid",
        ) >= SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN
        transition_y = np.flatnonzero(persistent_change)
        if len(transition_y) == 0:
            return ()

        # 같은 색 변화 구간에서 연속으로 나온 y는 첫 지점 하나만 남긴다.
        transition_starts = transition_y[
            np.r_[True, np.diff(transition_y) > 1]
        ]
        candidate_points.append(
            [(point_x, int(point_y + candidate_y)) for candidate_y in transition_starts]
        )

    if len(candidate_points) != 2:
        return ()

    valid_pairs = [
        (left_point, right_point)
        for left_point in candidate_points[0]
        for right_point in candidate_points[1]
        if abs(left_point[1] - right_point[1])
        < SOURCE_PREVIEW_INVALID_Y_DIFFERENCE
    ]
    if not valid_pairs:
        return ()

    # 가장 위에서 처음 만나는, y 차이가 가장 작은 쌍을 선택한다.
    return tuple(
        min(
            valid_pairs,
            key=lambda pair: (
                max(pair[0][1], pair[1][1]),
                abs(pair[0][1] - pair[1][1]),
            ),
        )
    )


def draw_top_pillar_reference_points(
    image, reference_points, downward_points=()
):
    """원본 A 페이지에 아래 방향 변화점 기준 수평선만 표시한다."""

    preview = to_bgr(image)
    if reference_points is None:
        return preview

    if len(downward_points) == 2:
        left_point, right_point = sorted(downward_points, key=lambda point: point[0])
        line_y = round((left_point[1] + right_point[1]) / 2)
        cv2.line(
            preview,
            (left_point[0], line_y),
            (right_point[0], line_y),
            PILLAR_DOWNWARD_POINT_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return preview


def create_polygon_preview(image, contours, measurements, mask_mode):
    """선택한 마스크 도형 내부의 원본 픽셀만 남긴 투명 Preview를 만든다."""

    tiles = []
    for contour, measurement in zip(contours, measurements):
        polygon = get_preview_mask_shape(contour, measurement, mask_mode)
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


def load_ab_tiff_pages(image_path):
    """A/B 다중 페이지 TIFF의 앞 두 페이지를 플랫폼과 무관하게 읽는다."""
    try:
        with Image.open(image_path) as tiff:
            if getattr(tiff, "n_frames", 1) < 2:
                raise ValueError("A/B 두 페이지 TIFF가 아닙니다.")

            pages = []
            for page_index in (0, 1):
                tiff.seek(page_index)
                page = np.array(tiff.copy())
                if page.ndim not in (2, 3):
                    raise ValueError(
                        f"{page_index + 1}번째 페이지의 차원이 올바르지 않습니다: "
                        f"{page.ndim}D"
                    )
                pages.append(page)
            return pages
    except (OSError, ValueError) as error:
        raise ValueError(f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}") from error


def create_detection_visualization(image_path):
    """B 분석선과 동일 좌표의 A 페이지 Crop 미리보기를 만든다."""

    image_a, image_b = load_ab_tiff_pages(image_path)
    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(f"A/B 페이지 크기가 일치하지 않습니다: {image_path}")

    gray = to_grayscale(image_b) # 이미지 파일 그레이 스케일 
    result = find_b_contours(image_b) # 컨투어링 작업

    for contour in result.contours:
        # 컨투어에 가장 먼저 닿는 접점 찾기
        measurement = find_first_contact_points(contour, gray.shape)
        result.measurements.append(measurement)

    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 그레이 스케일를 컬로로 변환하여 컨투어 색상이 보이게
    cv2.drawContours(result_image, result.contours, -1, CONTOUR_COLOR, 1)

    for measurement in result.measurements:
        # 방향별 첫 접점 분포의 대표점과 축 기준선만 방향색으로 표시한다.
        point_groups = [
            (measurement.top_points, TOP_COLOR, 1, True),
            (measurement.bottom_points, BOTTOM_COLOR, 1, False),
            (measurement.left_points, LEFT_COLOR, 0, True),
            (measurement.right_points, RIGHT_COLOR, 0, False),
        ]
        for points, color, coordinate_index, use_minimum in point_groups:
            reference_point = get_primary_contact_reference_point(
                points, coordinate_index, use_minimum
            )
            if reference_point is not None:
                draw_axis_aligned_reference_line(
                    result_image,
                    reference_point,
                    coordinate_index,
                    color,
                )
                display_points = [reference_point]
                spaced_point = get_spaced_reference_point(
                    points,
                    reference_point,
                    coordinate_index,
                    REFERENCE_POINT_MIN_DISTANCE,
                )
                if spaced_point is not None:
                    display_points.append(spaced_point)
                for display_point in display_points:
                    cv2.circle(
                        result_image,
                        display_point,
                        REFERENCE_POINT_RADIUS,
                        color,
                        thickness=cv2.FILLED,
                    )
                    cv2.circle(
                        result_image,
                        display_point,
                        REFERENCE_POINT_RADIUS + 1,
                        (0, 0, 0),
                        thickness=1,
                    )

    result.source_visualization = to_bgr(image_a)
    result.analysis_preview = create_polygon_preview(
        image_b,
        result.contours,
        result.measurements,
        ANALYSIS_PREVIEW_MASK_MODE,
    )
    result.source_preview = create_polygon_preview(
        image_a,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
    )
    return result_image, result

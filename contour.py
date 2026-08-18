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
BOTTOM_ROI_HEIGHT = 150
MIN_BOTTOM_CONTACT_X_DISTANCE = 60
MIN_BOTTOM_CONTACT_DISTANCE_FROM_TANGENT = 20
WHITE_PIXEL_THRESHOLD = 245

# 점(원)의 크기 조절
POINT_RADIUS = 2
EXTREME_POINT_RADIUS = 4
SECONDARY_POINT_RADIUS = 4
LINE_THICKNESS = 1        # 확장 직선 두께
EXTREME_SENCODARY_DIFF = 2
MIN_DISTANCE = 30
CROP_RECTANGLE_COLOR = (255, 255, 0)
CROP_RECTANGLE_THICKNESS = 1
BOTTOM_ROI_COLOR = (255, 255, 0)
BOTTOM_CONTACT_POINT_RADIUS = 4
CIRCLE_PREVIEW_OUTLINE_PADDING = 1
CIRCLE_PREVIEW_TILE_PADDING = 2
CIRCLE_PREVIEW_GAP = 6

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
    circle_preview: Optional[np.ndarray] = None


def to_grayscale(image):
    """이미지를 그레이스케일로 변환한다."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image):
    """시각화 선을 겹칠 수 있도록 이미지를 3채널 BGR로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
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


def get_crop_rectangle(measurement, image_shape):
    """하늘색 사각형과 Crop에 공통으로 사용할 좌표를 반환한다."""

    image_height, image_width = image_shape
    left = max(0, measurement.left_extreme[0])
    top = max(0, measurement.top_extreme[1])
    right = min(image_width - 1, measurement.right_extreme[0])
    bottom = min(image_height - 1, measurement.bottom_extreme[1])

    if left >= right or top >= bottom:
        return None

    return left, top, right, bottom


def crop_inside_rectangle(image_b, measurement):
    """하늘색 사각형 내부만 B 페이지에서 Crop한다."""

    crop_rectangle = get_crop_rectangle(measurement, image_b.shape[:2])
    if crop_rectangle is None:
        return None

    left, top, right, bottom = crop_rectangle
    cropped_image = to_grayscale(image_b)[top : bottom + 1, left : right + 1]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        cropped_image,
        (0, 0),
        (cropped_image.shape[1] - 1, cropped_image.shape[0] - 1),
        (0, 0, 0),
        1,
    )
    return cropped_image


def create_crop_preview(image_b, measurements):
    """하늘색 사각형 Crop 이미지를 사이드바 미리보기로 반환한다."""

    crops = []
    for measurement in measurements:
        cropped_image = crop_inside_rectangle(image_b, measurement)
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
        crop_height, crop_width = crop.shape[:2]
        offset_x = (preview_width - crop_width) // 2
        preview[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = crop
        offset_y += crop_height + gap

    return preview


def draw_bottom_reference_line(image, measurement):
    """하단 극단점 두 개로 만든 기준선을 한 페이지에 그린다."""

    draw_extended_line(
        image,
        measurement.bottom_extreme,
        measurement.bottom_secondary,
        BOTTOM_COLOR,
        LINE_THICKNESS,
    )


def draw_contact_points(image, points, color):
    """B 페이지와 동일한 크기와 색의 접점 점선을 그린다."""

    for point in points:
        cv2.circle(image, point, POINT_RADIUS, color, thickness=cv2.FILLED)


def get_line_y_at_x(pt1, pt2, point_x):
    """두 점으로 정의한 접선에서 지정한 x 좌표의 y 좌표를 반환한다."""

    if pt1 is None or pt2 is None:
        return None

    x1, y1 = pt1
    x2, y2 = pt2
    if x1 == x2:
        return round((y1 + y2) / 2)
    return round(y1 + (point_x - x1) * (y2 - y1) / (x2 - x1))


def get_bottom_tangent_span(measurement, image_shape):
    """하단 접점 점선이 놓인 접선의 좌우 범위와 양 끝 y 좌표를 반환한다."""

    if len(measurement.bottom_points) < 2:
        return None

    image_height, image_width = image_shape[:2]
    left_x = min(point[0] for point in measurement.bottom_points)
    right_x = max(point[0] for point in measurement.bottom_points)
    if left_x >= right_x:
        return None

    top_left_y = get_line_y_at_x(
        measurement.bottom_extreme, measurement.bottom_secondary, left_x
    )
    top_right_y = get_line_y_at_x(
        measurement.bottom_extreme, measurement.bottom_secondary, right_x
    )
    if top_left_y is None or top_right_y is None:
        return None

    return (
        int(np.clip(left_x, 0, image_width - 1)),
        int(np.clip(right_x, 0, image_width - 1)),
        int(np.clip(top_left_y, 0, image_height - 1)),
        int(np.clip(top_right_y, 0, image_height - 1)),
    )


def draw_bottom_roi(image, measurement):
    """하단 접선에서 아래로 150px인 ROI의 좌·우·하단 경계를 표시한다."""

    tangent_span = get_bottom_tangent_span(measurement, image.shape)
    if tangent_span is None:
        return

    left_x, right_x, top_left_y, top_right_y = tangent_span
    image_height = image.shape[0]
    bottom_left_y = min(image_height - 1, top_left_y + BOTTOM_ROI_HEIGHT)
    bottom_right_y = min(image_height - 1, top_right_y + BOTTOM_ROI_HEIGHT)
    cv2.line(image, (left_x, top_left_y), (left_x, bottom_left_y), BOTTOM_ROI_COLOR, 1)
    cv2.line(
        image,
        (left_x, bottom_left_y),
        (right_x, bottom_right_y),
        BOTTOM_ROI_COLOR,
        1,
    )
    cv2.line(image, (right_x, bottom_right_y), (right_x, top_right_y), BOTTOM_ROI_COLOR, 1)


def find_bottom_roi_contact_points(gray_image, measurement):
    """ROI 하단부터 위로 탐색해 조건을 만족하는 첫 접점 최대 3개를 찾는다."""

    tangent_span = get_bottom_tangent_span(measurement, gray_image.shape)
    if tangent_span is None:
        return []

    left_x, right_x, _, _ = tangent_span
    image_height = gray_image.shape[0]
    candidates = []
    for point_x in range(left_x, right_x + 1):
        tangent_y = get_line_y_at_x(
            measurement.bottom_extreme, measurement.bottom_secondary, point_x
        )
        if tangent_y is None:
            continue

        roi_top_y = int(np.clip(tangent_y, 0, image_height - 1))
        roi_bottom_y = min(image_height - 1, roi_top_y + BOTTOM_ROI_HEIGHT)
        column = gray_image[roi_top_y : roi_bottom_y + 1, point_x]
        non_white_offsets = np.where(column < WHITE_PIXEL_THRESHOLD)[0]
        if len(non_white_offsets) == 0:
            continue

        # ROI 가장 밑에서 위로 올라갈 때 처음 닿는 비백색 픽셀이다.
        contact_y = roi_top_y + int(non_white_offsets[-1])
        if contact_y - tangent_y < MIN_BOTTOM_CONTACT_DISTANCE_FROM_TANGENT:
            continue
        candidates.append((point_x, contact_y))

    candidates.sort(key=lambda point: (-point[1], point[0]))
    contact_points = []
    for candidate in candidates:
        if all(
            abs(candidate[0] - contact[0]) >= MIN_BOTTOM_CONTACT_X_DISTANCE
            for contact in contact_points
        ):
            contact_points.append(candidate)
        if len(contact_points) == 3:
            break
    return sorted(contact_points)


def get_bottom_contact_circle(measurement, point):
    """접점과 동일 x좌표의 접선이 만드는 원의 중심과 반지름을 반환한다."""

    tangent_y = get_line_y_at_x(
        measurement.bottom_extreme, measurement.bottom_secondary, point[0]
    )
    if tangent_y is None:
        return None

    diameter = point[1] - tangent_y
    if diameter <= 0:
        return None
    return (point[0], round((tangent_y + point[1]) / 2)), round(diameter / 2)


def draw_bottom_roi_contact_points(image, measurement, contact_points):
    """접점과 접선 사이 거리를 지름으로 하는 원과 접점을 표시한다."""

    for point in contact_points:
        circle = get_bottom_contact_circle(measurement, point)
        if circle is not None:
            center, radius = circle
            cv2.circle(image, center, radius, BOTTOM_COLOR, thickness=1)

        cv2.circle(
            image,
            point,
            BOTTOM_CONTACT_POINT_RADIUS,
            EXTREME_POINT_COLOR,
            thickness=cv2.FILLED,
        )
        cv2.circle(
            image,
            point,
            BOTTOM_CONTACT_POINT_RADIUS + 1,
            (0, 0, 0),
            thickness=1,
        )


def create_circle_preview(image_a, measurements):
    """노란 접점 없이 원 내부 픽셀과 빨간 원 테두리만 미리보기로 만든다."""

    source_gray = to_grayscale(image_a)
    circles = []
    for measurement in measurements:
        contact_points = find_bottom_roi_contact_points(source_gray, measurement)
        for point in contact_points:
            circle = get_bottom_contact_circle(measurement, point)
            if circle is not None:
                circles.append(circle)

    if not circles:
        return None

    source_image = to_bgr(image_a)
    image_height, image_width = source_image.shape[:2]
    circle_tiles = []
    for (center_x, center_y), radius in sorted(circles):
        extent = radius + CIRCLE_PREVIEW_OUTLINE_PADDING + CIRCLE_PREVIEW_TILE_PADDING
        left = max(0, center_x - extent)
        top = max(0, center_y - extent)
        right = min(image_width - 1, center_x + extent)
        bottom = min(image_height - 1, center_y + extent)

        tile_source = source_image[top : bottom + 1, left : right + 1].copy()
        tile_mask = np.zeros(tile_source.shape[:2], dtype=np.uint8)
        local_center = (center_x - left, center_y - top)
        cv2.circle(tile_source, local_center, radius, BOTTOM_COLOR, thickness=1)
        cv2.circle(
            tile_mask,
            local_center,
            radius + CIRCLE_PREVIEW_OUTLINE_PADDING,
            255,
            thickness=cv2.FILLED,
        )

        tile = np.full_like(tile_source, 255)
        tile[tile_mask == 255] = tile_source[tile_mask == 255]
        circle_tiles.append(tile)

    preview_height = max(tile.shape[0] for tile in circle_tiles)
    preview_width = sum(tile.shape[1] for tile in circle_tiles)
    preview_width += CIRCLE_PREVIEW_GAP * (len(circle_tiles) - 1)
    preview = np.full((preview_height, preview_width, 3), 255, dtype=np.uint8)

    offset_x = 0
    for tile in circle_tiles:
        offset_y = (preview_height - tile.shape[0]) // 2
        tile_height, tile_width = tile.shape[:2]
        preview[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width] = tile
        offset_x += tile_width + CIRCLE_PREVIEW_GAP
    return preview


def create_a_visualization(image_a, measurements):
    """A 페이지에 B의 하단 점선, ROI와 ROI 하단 접점을 표시한다."""

    source_image = to_bgr(image_a)
    source_gray = to_grayscale(image_a)
    for measurement in measurements:
        # A/B 페이지는 같은 크기와 좌표계를 공유하므로 B의 하단 점선을 그대로 사용한다.
        draw_bottom_roi(source_image, measurement)
        draw_contact_points(source_image, measurement.bottom_points, BOTTOM_COLOR)
        contact_points = find_bottom_roi_contact_points(source_gray, measurement)
        draw_bottom_roi_contact_points(source_image, measurement, contact_points)
    return source_image


def create_detection_visualization(image_path):
    """A/B 페이지 시각화와 B 페이지 컨투어 분석 결과를 만든다."""

    success, images = cv2.imreadmulti(str(image_path), flags=cv2.IMREAD_UNCHANGED)
    if not success or len(images) < 2:
        raise ValueError(f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}")

    image_a = images[0]
    image_b = images[1]
    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(f"A/B 페이지 크기가 일치하지 않습니다: {image_path}")

    gray = to_grayscale(image_b) # 이미지 파일 그레이 스케일 
    result = find_b_contours(image_b) # 컨투어링 작업

    for contour in result.contours:
        # 컨투어에 가장 먼저 닿는 접점 찾기
        measurement = find_first_contact_points(contour, gray.shape)
        result.measurements.append(measurement)

    source_image = create_a_visualization(image_a, result.measurements)
    result.circle_preview = create_circle_preview(image_a, result.measurements)
    result.crop_preview = create_crop_preview(image_b, result.measurements) # 사이드바 프리뷰
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 그레이 스케일를 컬로로 변환하여 컨투어 색상이 보이게
    cv2.drawContours(result_image, result.contours, -1, CONTOUR_COLOR, 1) # 컨투어 그리기 

    for measurement in result.measurements:
        # 1. 상·하·좌·우 두 점 연결 직선 그리기 (이미지 끝까지 연장)
        draw_extended_line(result_image, measurement.top_extreme, measurement.top_secondary, TOP_COLOR, LINE_THICKNESS)
        draw_bottom_reference_line(result_image, measurement)
        draw_extended_line(result_image, measurement.left_extreme, measurement.left_secondary, LEFT_COLOR, LINE_THICKNESS)
        draw_extended_line(result_image, measurement.right_extreme, measurement.right_secondary, RIGHT_COLOR, LINE_THICKNESS)

        crop_rectangle = get_crop_rectangle(measurement, gray.shape)
        if crop_rectangle is not None:
            left, top, right, bottom = crop_rectangle
            cv2.rectangle(
                result_image,
                (left, top),
                (right, bottom),
                CROP_RECTANGLE_COLOR,
                CROP_RECTANGLE_THICKNESS,
            )

        # 2. 샘플 점들 그리기
        point_groups = [
            (measurement.top_points, TOP_COLOR),
            (measurement.bottom_points, BOTTOM_COLOR),
            (measurement.left_points, LEFT_COLOR),
            (measurement.right_points, RIGHT_COLOR),
        ]
        for points, color in point_groups:
            draw_contact_points(result_image, points, color)

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

    return source_image, result_image, result

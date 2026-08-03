
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import cv2
import numpy as np

# ------------ Data Class ----------- #
@dataclass
class SurfaceDetection:
    x: int
    y: int
    width: int
    height: int
    
@dataclass
class BallDetection:
    x: int
    y: int
    radius: int

@dataclass
class DetectionResult:
    surface: Optional[SurfaceDetection] = None
    balls: List[BallDetection] = field(default_factory=list)



# ------------ Data Class ----------- #

SURFACE_COLOR = (80, 255, 80) # GREEN(BGR)
BALL_COLOR = (0, 0, 255) # RED(BGR)
SURFACE_LINE_THICKNESS = 1
LEFT_ZONE_MAX = 0.35
RIGHT_ZONE_MIN = 0.65



def contour_visualization(image_path: Path, detect_surface=True, detect_balls=True):
    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) # 그레이 스케일(ndarray)
    if gray_image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    detected_image = detect_shapes(gray_image, detect_surface, detect_balls)
    return draw_detections(gray_image, detected_image), detected_image

def create_detection_visualization(
    image_path: Path,
    detect_surface=True,
    detect_balls=True,
):
    """app.py가 호출하는 Surface/Ball 컨투어 검출 함수."""
    return contour_visualization(
        image_path,
        detect_surface=detect_surface,
        detect_balls=detect_balls,
    )

def draw_detections(gray: np.ndarray, result:DetectionResult):
    """이미지에 컨투어 라인을 그리는 함수"""
    visualized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Surface
    if result.surface is not None:
        surface = result.surface
        cv2.rectangle(
            visualized,
            (surface.x, surface.y),
            (surface.x + surface.width, surface.y + surface.height),
            SURFACE_COLOR,
            SURFACE_LINE_THICKNESS, # 두께
        )
    
    for i, ball in enumerate(result.balls, start=1):
        cv2.circle(
            visualized,
            (ball.x, ball.y),
            ball.radius,
            BALL_COLOR,
            2, # 굵기
        )
    return visualized

def detect_shapes(gray: np.ndarray, detect_surface=True, detect_balls=True):
    """### 도형을 탐지하는 함수
    - 원을 탐지하려면 사각형 표면을 먼저 탐지해야함.
    - 그래서 원만 탐지하려면 탐지 코드를 작동하되 결과에서는 출력하지 않음.
    """

    # Detecting을 하지 않는 상태
    if not detect_surface and not detect_balls:
        return DetectionResult(surface=None, balls=[])

    base_surface = find_surface_contour(gray)
    balls = (
        find_balls_contours(gray, base_surface)
        if detect_balls and base_surface is not None else []
    )
    surface = base_surface if detect_surface else None

    return DetectionResult(surface=surface, balls=balls)


def select_balls_by_layout(
    candidates, surface_left, surface_width):
    """Ball 후보의 배치를 확인해 최종 검출 결과를 반환한다.

    Ball이 0개면 빈 목록을, 1개면 해당 후보 하나를 그대로 반환한다.
    2개 이상일 때는 기존의 좌·중앙·우 배치 규칙을 적용한다.
    """
    if len(candidates) <= 1:
        return sorted(candidates, key=lambda ball: ball.x)

    left_balls: List[BallDetection] = []
    center_balls: List[BallDetection] = []
    right_balls: List[BallDetection] = []

    for ball in candidates:
        relative_x = (ball.x - surface_left) / surface_width
        if relative_x <= LEFT_ZONE_MAX:
            left_balls.append(ball)
        elif relative_x >= RIGHT_ZONE_MIN:
            right_balls.append(ball)
        else:
            center_balls.append(ball)

    # 볼이 두개인 경우
    is_two_ball_layout = (
        len(left_balls) == 1
        and len(center_balls) == 0
        and len(right_balls) == 1
    )

    # 볼이 세개인 경우
    is_three_ball_layout = (
        len(left_balls) == 1
        and len(center_balls) == 1
        and len(right_balls) == 1
    )

    if is_two_ball_layout or is_three_ball_layout:
        return sorted(candidates, key=lambda ball: ball.x)
    return []


def find_surface_contour(gray: np.ndarray):
    """### 사각형 표면을 탐지하는 함수 
    1. 마스크 생성 (밝기 235 이상 -> 검정, 235 미만 -> 흰색)
    2. 모폴로지 opening 연산으로 노이즈 제거 및 표면만 남기기
    3. 바깥 윤곽선만 찾음
    4. 후보군 필터링
    5. 가장 큰 박스를 표면으로 선택
    """
    height, width = gray.shape
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV) # 마스크 생성  (밝기 235 이상 -> 검정, 235 미만 -> 흰색)

    # 마스크 구조 작업
    horizontal_kernel_width = max(3,round(width * 0.333)) # 전체 이미지의 33.3% 크기의 가로 (최소 3픽셀 이상)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width,5)) # 가로 세로(3px) 크기의 커널을 남김  (직사각형, (크기))
    surface_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel) # 모폴로지 opening 연산으로 노이즈 제거 및 표면만 남기기

    # 정리된 마스크에서 윤곽선 찾고 후보군 필터링
    contours, _ = cv2.findContours(surface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 바깥 윤곽선만 찾음
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = box_width / max(box_height, 1)

        is_wide_enough = box_width >= width * 0.3 # 박스 너비가 이미지 전체 너비의 30% 이상인지
        is_lower_part = y >= height * 0.25 # y좌표가 이미지 전체 높이의 25% 지점보다 아래인지?
        is_surface_like = aspect_ratio >= 2.5 # 가로가 세로보다 2.5배 이상인지

        if is_wide_enough and is_lower_part and is_surface_like:
            # 3가지 조건 모두 만족하면 박스의 너비를 점수로 산정해 후보 목록에 담는다.
            score = box_width * box_height
            candidates.append((score, (x, y, box_width, box_height)))

    if not candidates:
        return None

    _, (x, y, box_width, box_height) = max(candidates, key=lambda item: item[0]) # 면적이 가장 큰 박스를 택해 반환
    return SurfaceDetection(x, y, box_width, box_height)


def find_balls_contours(gray: np.ndarray, surface: SurfaceDetection):
    """### 볼의 위치를 탐지해내는 함수
    1. Surface의 Contour를 찾아냄
    2. ROI(관심영역) 추출
    3. 블러링으로 노이즈 제거
    4. 허프 변환으로 원 탐지
    5. 볼들의 위치 반환
    
    """
    image_height, _ = gray.shape
    x, y, surface_width, surface_height = surface.x, surface.y, surface.width, surface.height
    
    # ROI 설정 (y + surface_height : 직사각형 표면 아래 영역)
    roi_top = max(0, y + surface_height - round(image_height * 0.03)) #12px (400 x 0.03), 즉 surface_bottom 보다 12px 위에서 부터 시작
    roi_bottom = min(image_height, y + surface_height + round(image_height * 0.08)) #32px (400 x 0.08), 즉 surface_bottom 보다 32px 아래로
    cropped_roi = gray[roi_top:roi_bottom, x : x + surface_width] # 볼 위치 부분만 잘라냄
    blurred_roi = cv2.GaussianBlur(cropped_roi, (5, 5), 0) # 5x5 크기의 가우시안 필터로 노이즈 제거해서 큰 흰색 표면의 굴곡이 윤곽선으로 잡히는 것을 방지함

    # 허프 변환을 이용한 원탐지
    detected_circles = cv2.HoughCircles(
        blurred_roi, # 볼이 있는 이미지 부분
        cv2.HOUGH_GRADIENT,
        dp = 1.0, # 내부 해상도
        minDist = round(surface_width * 0.16), # 원의 중심과 최소거리
        param1 = 50, # 약한 테두리 무시되는 정도 (높을수록 확실한 테두리만 탐지)
        param2 = 12, # 원이라고 판단할 최소한의 점수 (낮을수록 원 탐지에 관대해짐)
        minRadius = max(4, round(image_height * 0.01)), # 허용 최소 반지름
        maxRadius = round(image_height * 0.04), # 하용 최대 반지름
    )

    if detected_circles is None:
        return []
    
    candidates: List[BallDetection] = []
    for center_x, center_y, radius in np.round(detected_circles[0]).astype(int):
        candidates.append(
            BallDetection(
                int(x + center_x), int(roi_top + center_y), int(radius)
            )
        )

    return select_balls_by_layout(candidates, x, surface_width)


    

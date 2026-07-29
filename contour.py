
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
            2, # 굵기
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


def find_surface_contour(gray: np.ndarray):
    """ 사각형 표면을 탐지하는 함수 """
    height, width = gray.shape
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV) # 마스크 생성  

    # 마스크 구조 작업
    horizontal_kernel_width = max(3,round(width * 0.333)) # 전체 이미지의 33.3% 크기의 가로 (최소 3픽셀 이상)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width,3)) # 가로 세로(3px) 크기의 커널
    surface_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel) # 모폴로지 opening 연산으로 노이즈 제거 및 표면만 남기기

    # 윤곽선 찾고 후보군 필터링
    contours, _ = cv2.findContours(surface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = box_width / max(box_height, 1)

        is_wide_enough = box_width >= width * 0.3 # 박스 너비가 이미지 전체 너비의 30% 이상인지
        is_lower_part = y >= height * 0.25 # y좌표가 이미지 전체 높이의 25% 지점보다 아래인지?
        is_surface_like = aspect_ratio >= 2.5 # 가로 세로 비율이 2.5 이상인지

        if is_wide_enough and is_lower_part and is_surface_like:
            # 3가지 조건 모두 만족하면 박스의 너비를 점수로 산정해 후보 목록에 담는다.
            score = box_width * box_height
            candidates.append((score, (x, y, box_width, box_height)))

    if not candidates:
        return None

    _, (x, y, box_width, box_height) = max(candidates, key=lambda item: item[0]) # 면적이 가장 큰 박스를 택해 반환
    return SurfaceDetection(x, y, box_width, box_height)


def find_balls_contours(gray: np.ndarray, surface: SurfaceDetection):
    """볼의 위치를 탐지해내는 함수"""
    image_height, _ = gray.shape
    x, y, surface_width, surface_height = surface.x, surface.y, surface.width, surface.height
    
    # ROI 설정 (직사각형 표면 아래 영역)
    roi_top = max(0, y + surface_height - round(image_height * 0.04))
    roi_bottom = min(image_height, y + surface_height + round(image_height * 0.08))
    wheel_roi = gray[roi_top:roi_bottom, x : x + surface_width]
    blurred_roi = cv2.GaussianBlur(wheel_roi, (5, 5), 0) # 5x5 크기의 가우시안 필터로 노이즈 제거해서 큰 흰색 표면의 굴곡이 윤곽선으로 잡히는 것을 방지함

    # 허프 변환을 이용한 원탐지
    detected_circles = cv2.HoughCircles(
        blurred_roi,
        cv2.HOUGH_GRADIENT,
        dp = 1.0,
        minDist = round(surface_width * 0.16), # 원들 사이의 최소 거리
        param1 = 50,
        param2 = 12, # 원이라고 판단할 최소한의 점수
        minRadius = max(4, round(image_height * 0.01)), # 최소 반지름
        maxRadius = round(image_height * 0.04), # 최대 반지름
    )
    if detected_circles is None:
        return []
    
    detections: list[BallDetection] = []
    for center_x, center_y, radius in np.round(detected_circles[0]).astype(int):
        # 볼의 상대적 위치 계산
        relative_x = center_x / surface_width
        is_near_an_edge = relative_x <= 0.35 or relative_x >= 0.65 # 직사각형 표면 아래 좌측 35%, 우측 35%에 있는 원만 바퀴로 인정
        if is_near_an_edge:
            detections.append(
                BallDetection(
                    int(x + center_x), int(roi_top + center_y), int(radius)
                )
            )
    
    return sorted(detections, key=lambda ball:ball.x)


    

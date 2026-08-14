# Die Surface Contour Inspector

> An OpenCV-based program designed to swiftly detect defects on the top, bottom, left, and right plates of a die surface.

<img src="assets/thumnail.png" width=1280>

## Current Feature
#### Side Surface
1. Processed the side images using brightness binarization and contouring, generating vertical search lines based on the initial contact points from the image edges to the surface.
     - To prevent errors caused by a slight rotation of the side, the surface was divided into ‭$N$‬ segments to accurately identify the contact points.
2. For the side balls, the bottom section of the surface contour was divided into three parts to find the furthest point, and the process of fitting a circle centered on that point is currently in progress.

#### Top-Bottom Surface
_[Upcoming]_


## Project Structure

```text

├── app.py             # Starts the PyQt5 application
├── ui.py              # Displays A- and B-page results together 
├── contour.py         # B-page contour 
├── styles.py          # UI styles (created by Claude)
├── requirements.txt   # Python dependencies
├── assets/            # Application icon and other assets
└── dataset/            

```

## Requirements

- Python 3.10–3.12 recommended
- Input `TIFF` images must contain A/B pages with identical dimensions (specific Dataset)

## Installation

Run the following commands after receiving the project.

### macOS / Linux

```bash
cd diehand_cv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd diehand_cv
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

```bash
python app.py
```

> Note: Since this repository is a program designed for specific experimental research, it may not be suitable for your project.

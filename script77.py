"""棒球 3D 姿態分析 + 生物力學曲線 + Gemini 診斷 (Tasks API 版)
執行：streamlit run script77.py
"""
import json
import os
import tempfile
import urllib.request
from typing import List

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
import streamlit as st

# Google GenAI 套件
from google import genai
from google.genai import types

st.set_page_config(page_title="棒球 3D 生物力學診斷系統", page_icon="⚾", layout="wide")

BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28)
]
JOINTS = sorted(set(i for pair in BONES for i in pair))

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker_heavy.task"


# ─── 1. Pydantic 診斷報告資料模型 ─────────────────────────────────
class DrillRecommendation(BaseModel):
    drill_name: str = Field(description="輔助訓練名稱")
    objective: str = Field(description="此訓練欲改善的動作瑕疵")
    execution: str = Field(description="具體執行步驟與注意事項")


class BiomechanicalIssue(BaseModel):
    issue_name: str = Field(description="問題名稱，例如：X-Factor 儲能不足、前膝支撐吸收能量等")
    severity: str = Field(description="嚴重程度: 低 / 中 / 高")
    analysis: str = Field(description="詳細生物力學分析與對揮棒/投球表現或受傷風險的影響")


class BaseballAnalysisReport(BaseModel):
    overall_score: int = Field(description="動作機制綜合評分 (1-100)")
    summary: str = Field(description="整體動作評價總結")
    peak_x_factor_eval: str = Field(description="對最大髖肩分離度 (X-Factor) 的力學評語")
    wrist_speed_eval: str = Field(description="對手腕加速曲線與峰值速度的評語")
    detected_issues: List[BiomechanicalIssue] = Field(description="檢測到的動作瑕疵列表")
    recommended_drills: List[DrillRecommendation] = Field(description="推薦的矯正訓練菜單")


# ─── 2. 模型檔案檢查 ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("首次執行，正在自動下載 MediaPipe 3D 姿態模型..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# ─── 3. 運動生物力學幾何計算函式 ─────────────────────────────────
def calculate_angle_3d(a, b, c):
    """計算三點 a-b-c 的 3D 夾角 (度)"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_x_factor(pts):
    """計算 3D 空間中髖連線向量與肩膀連線向量在 XY 平面的投影夾角 (X-Factor)"""
    hip_vec = pts[24] - pts[23]  # 右髖 - 左髖
    shoulder_vec = pts[12] - pts[11]  # 右肩 - 左肩

    # 投影到 XY 水平平面
    hip_vec_2d = hip_vec[[0, 1]]
    shoulder_vec_2d = shoulder_vec[[0, 1]]

    unit_hip = hip_vec_2d / (np.linalg.norm(hip_vec_2d) + 1e-6)
    unit_shoulder = shoulder_vec_2d / (np.linalg.norm(shoulder_vec_2d) + 1e-6)

    dot_prod = np.dot(unit_hip, unit_shoulder)
    angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    return angle


# ─── 4. 影片 3D 擷取與特徵數據計算 ─────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=3)
def analyze_video(video_bytes, suffix, sample_step):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_file = ensure_model_file()
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    path, cap, records = None, None, []
    fps = 30.0

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(video_bytes)
            path = handle.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("影片無法開啟，請確認編碼格式（建議 H.264 MP4）。")

        read_fps = cap.get(cv2.CAP_PROP_FPS)
        if np.isfinite(read_fps) and read_fps > 0:
            fps = float(read_fps)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0, text="分析影片與生物力學數據中…")

        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if index % sample_step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int((index / fps) * 1000)

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                        landmarks = result.pose_world_landmarks[0]
                        pts = np.array([[p.x, p.z, -p.y] for p in landmarks], dtype=float)
                        if pts.shape == (33, 3) and np.isfinite(pts).all():
                            records.append((index, pts))

                index += 1
                if total > 0 and index % 15 == 0:
                    progress.progress(min(index / total, 1.0), text=f"已處理 {index} 格")
        finally:
            landmarker.close()
            progress.empty()

        if not records:
            raise ValueError("影片中未讀取到有效的骨架影格。")

        return records, fps, index
    finally:
        if cap is not None:
            cap.release()
        if path and os.path.exists(path):
            os.remove(path)


# ─── 5. 計算時間軸生物力學指標 (Metrics Timeline) ─────────────────
def compute_biomechanics_timeline(records, fps, tracked_joint):
    frame_indices = [r[0] for r in records]
    time_sec = np.array(frame_indices) / fps
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    poses = np.stack([pts for _, pts in records])
    hip_centers = (poses[:, 23, :] + poses[:, 24, :]) / 2
    poses_rel = poses - hip_centers[:, None, :]

    # 1. 計算 X-Factor 時間序列
    x_factors = [calculate_x_factor(pts) for pts in poses]

    # 2. 計算前膝支撐角度 (以左膝/右膝為例，取右腳24-26-28與左腳23-25-27)
    knee_angles = [calculate_angle_3d(pts[23], pts[25], pts[27]) for pts in poses]

    # 3. 計算指定關節 (如手腕) 的 3D 移動速度 (m/s)
    traj = poses_rel[:, tracked_joint, :]
    vel_vectors = np.diff(traj, axis=0) / dt
    speeds = np.linalg.norm(vel_vectors, axis=1)
    speeds = np.insert(speeds, 0, 0.0)  # 第一格補零

    # 轉換為 mph (1 m/s ≈ 2.23694 mph)
    speeds_mph = speeds * 2.23694

    metrics_data = {
        "time_sec": time_sec,
        "frame_indices": frame_indices,
        "x_factors": x_factors,
        "knee_angles": knee_angles,
        "speeds_mph": speeds_mph,
        "max_x_factor": float(np.max(x_factors)),
        "max_wrist_speed_mph": float(np.max(speeds_mph)),
        "avg_knee_angle": float(np.mean(knee_angles))
    }
    return metrics_data, poses_rel

# ─── 6. 繪製生物力學折線圖 ─────────────────────────────────────────
def make_metrics_chart(metrics_data):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("髖肩分離度 (X-Factor Sequence)", "追蹤關節 3D 移動速度 (Wrist/Elbow Speed)")
    )

    # X-Factor 軌跡
    fig.add_trace(
        go.Scatter(
            x=metrics_data["time_sec"],
            y=metrics_data["x_factors"],
            mode="lines+markers",
            name="X-Factor (度)",
            line=dict(color="#00d8d6", width=3)
        ),
        row=1, col=1
    )

    # 速度軌跡
    fig.add_trace(
        go.Scatter(
            x=metrics_data["time_sec"],
            y=metrics_data["speeds_mph"],
            mode="lines+markers",
            name="速度 (mph)",
            line=dict(color="#ffdd59", width=3)
        ),
        row=2, col=1
    )

    fig.update_layout(
        paper_bgcolor="#1e272e",
        plot_bgcolor="#2d3436",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    fig.update_xaxes(title_text="時間 (秒)", gridcolor="#485460", row=2, col=1)
    fig.update_yaxes(title_text="角度 (°)", gridcolor="#485460", row=1, col=1)
    fig.update_yaxes(title_text="速度 (mph)", gridcolor="#485460", row=2, col=1)

    return fig
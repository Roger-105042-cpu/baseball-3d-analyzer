"""
棒球 3D 生物力學與 AI 診斷系統 (單次 & 多次揮棒 3D 建模版)
執行方式: streamlit run script77.py
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

# 1. 頁面初始化 (必須為第一個 Streamlit 命令)
st.set_page_config(page_title="棒球 3D 生物力學診斷系統", page_icon="⚾", layout="wide")

# Google GenAI 套件檢查
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28)
]
JOINTS = sorted(set(i for pair in BONES for i in pair))

# 使用 Lite 輕量化模型
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"


# ─── Pydantic 診斷報告資料結構 ───────────────────────────────────────
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


# ─── 模型下載與快取 ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("首次執行，正在自動下載輕量化 3D 姿態估計模型..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# ─── 幾何與向量計算 ───────────────────────────────────────────────
def calculate_angle_3d(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_x_factor(pts):
    hip_vec = pts[24] - pts[23]
    shoulder_vec = pts[12] - pts[11]
    hip_vec_2d = hip_vec[[0, 1]]
    shoulder_vec_2d = shoulder_vec[[0, 1]]
    unit_hip = hip_vec_2d / (np.linalg.norm(hip_vec_2d) + 1e-6)
    unit_shoulder = shoulder_vec_2d / (np.linalg.norm(shoulder_vec_2d) + 1e-6)
    return np.degrees(np.arccos(np.clip(np.dot(unit_hip, unit_shoulder), -1.0, 1.0)))


# ─── 影片分析與姿態擷取 ───────────────────────────────────────────
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
        progress = st.progress(0, text="正在分析 3D 姿態與關節特徵…")

        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if index % sample_step == 0:
                    h, w = frame.shape[:2]
                    if w > 640:
                        new_h = int(h * (640 / w))
                        frame = cv2.resize(frame, (640, new_h), interpolation=cv2.INTER_NEAREST)

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
                if total > 0 and index % 30 == 0:
                    progress.progress(min(index / total, 1.0), text=f"已加速處理 {index}/{total} 格")
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


# ─── 時間軸生物力學指標計算 ───────────────────────────────────────
def compute_biomechanics_timeline(records, fps, tracked_joint):
    frame_indices = [r[0] for r in records]
    time_sec = np.array(frame_indices) / fps
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    poses = np.stack([pts for _, pts in records])
    hip_centers = (poses[:, 23, :] + poses[:, 24, :]) / 2
    poses_rel = poses - hip_centers[:, None, :]

    x_factors = [calculate_x_factor(pts) for pts in poses]
    knee_angles = [calculate_angle_3d(pts[23], pts[25], pts[27]) for pts in poses]

    traj = poses_rel[:, tracked_joint, :]
    vel_vectors = np.diff(traj, axis=0) / dt
    speeds = np.linalg.norm(vel_vectors, axis=1)
    speeds = np.insert(speeds, 0, 0.0)
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


# ─── 自動揮棒事件分割 (Swings Segmentation) ──────────────────────────
def segment_swings(metrics_data, fps, window_sec=1.0):
    speeds = metrics_data["speeds_mph"]
    times = metrics_data["time_sec"]

    threshold = np.mean(speeds) + 1.2 * np.std(speeds)
    peak_indices = []

    for i in range(1, len(speeds) - 1):
        if speeds[i] > threshold and speeds[i] > speeds[i - 1] and speeds[i] > speeds[i + 1]:
            if not peak_indices or (times[i] - times[peak_indices[-1]]) > 1.2:
                peak_indices.append(i)

    if not peak_indices:
        return [{"name": "🌐 全景整段影片 (全揮棒)", "start_idx": 0, "end_idx": len(times) - 1, "peak_idx": int(np.argmax(speeds))}]

    window_frames = int(window_sec * fps)
    swings = [{"name": "🌐 全景整段影片 (含多次揮棒)", "start_idx": 0, "end_idx": len(times) - 1, "peak_idx": int(np.argmax(speeds))}]

    for idx, p_idx in enumerate(peak_indices):
        start_idx = max(0, p_idx - int(window_frames * 0.4))
        end_idx = min(len(times) - 1, p_idx + int(window_frames * 0.6))
        peak_time = times[p_idx]
        swings.append({
            "name": f"🎯 單次揮棒 #{idx + 1} (擊球峰值時間: {peak_time:.2f}秒)",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "peak_idx": p_idx
        })

    return swings


# ─── 繪製 2D 運動力學曲線 ─────────────────────────────────────────
def make_metrics_chart(metrics_data):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("髖肩分離度 (X-Factor Sequence)", "追蹤關節 3D 移動速度 (Speed MPH)")
    )
    fig.add_trace(
        go.Scatter(x=metrics_data["time_sec"], y=metrics_data["x_factors"], mode="lines+markers", name="X-Factor (°)", line=dict(color="#00d8d6", width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=metrics_data["time_sec"], y=metrics_data["speeds_mph"], mode="lines+markers", name="速度 (mph)", line=dict(color="#ffdd59", width=3)),
        row=2, col=1
    )
    fig.update_layout(
        paper_bgcolor="#1e272e", plot_bgcolor="#2d3436", font=dict(color="#ffffff"),
        height=400, margin=dict(l=20, r=20, t=40, b=20), showlegend=False
    )
    fig.update_xaxes(title_text="時間 (秒)", gridcolor="#485460", row=2, col=1)
    fig.update_yaxes(title_text="角度 (°)", gridcolor="#485460", row=1, col=1)
    fig.update_yaxes(title_text="速度 (mph)", gridcolor="#485460", row=2, col=1)
    return fig


# ─── 繪製 3D 骨架與球棒揮擊扇形面 (Bat Swept Surface) ───────────────
def make_3d_figure(records, poses_rel, tracked_joint, animation_limit=240, bat_length=0.85):
    stride = max(1, int(np.ceil(len(records) / animation_limit)))
    anim_records = records[::stride]
    anim_poses = poses_rel[::stride]

    wrist_idx = 16 if tracked_joint in [16, 14] else 15
    elbow_idx = 14 if tracked_joint in [16, 14] else 13

    wrist_traj = poses_rel[:, wrist_idx, :]
    elbow_traj = poses_rel[:, elbow_idx, :]

    bat_vecs = wrist_traj - elbow_traj
    bat_vecs = bat_vecs / (np.linalg.norm(bat_vecs, axis=1, keepdims=True) + 1e-6)
    tip_traj = wrist_traj + bat_vecs * bat_length

    num_frames = len(wrist_traj)
    num_segments = 10
    s = np.linspace(0, 1, num_segments)
    surf_X = np.zeros((num_frames, num_segments))
    surf_Y = np.zeros((num_frames, num_segments))
    surf_Z = np.zeros((num_frames, num_segments))

    for t in range(num_frames):
        for idx_s, val_s in enumerate(s):
            pt = (1 - val_s) * wrist_traj[t] + val_s * tip_traj[t]
            surf_X[t, idx_s] = pt[0]
            surf_Y[t, idx_s] = pt[1]
            surf_Z[t, idx_s] = pt[2]

    fig = go.Figure()

    # 3D 半透明揮棒扇形面
    fig.add_trace(go.Surface(
        x=surf_X, y=surf_Y, z=surf_Z,
        colorscale="Viridis", opacity=0.6, showscale=False, name="揮棒軌跡扇形面"
    ))
    fig.add_trace(go.Scatter3d(x=wrist_traj[:, 0], y=wrist_traj[:, 1], z=wrist_traj[:, 2], mode="lines", line=dict(color="#00d8d6", width=3), name="手腕軌跡"))
    fig.add_trace(go.Scatter3d(x=tip_traj[:, 0], y=tip_traj[:, 1], z=tip_traj[:, 2], mode="lines", line=dict(color="#ffdd59", width=4), name="棒頭軌跡 (扇形外緣)"))

    init_pose = anim_poses[0]
    init_joints = init_pose[JOINTS]
    fig.add_trace(go.Scatter3d(x=init_joints[:, 0], y=init_joints[:, 1], z=init_joints[:, 2], mode="markers", marker=dict(size=5, color="#ff5e57"), name="關節點"))

    for p1_idx, p2_idx in BONES:
        p1, p2 = init_pose[p1_idx], init_pose[p2_idx]
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode="lines", line=dict(color="#ffffff", width=4), showlegend=False))

    anim_frames = []
    anim_wrist = wrist_traj[::stride]
    anim_tip = tip_traj[::stride]

    for idx, (frame_data, pose_pts) in enumerate(zip(anim_records, anim_poses)):
        curr_joints = pose_pts[JOINTS]
        w_pt = anim_wrist[idx]
        t_pt = anim_tip[idx]

        frame_traces = [
            go.Surface(x=surf_X, y=surf_Y, z=surf_Z, colorscale="Viridis", opacity=0.6, showscale=False),
            go.Scatter3d(x=wrist_traj[:, 0], y=wrist_traj[:, 1], z=wrist_traj[:, 2], mode="lines", line=dict(color="#00d8d6", width=3)),
            go.Scatter3d(x=tip_traj[:, 0], y=tip_traj[:, 1], z=tip_traj[:, 2], mode="lines", line=dict(color="#ffdd59", width=4)),
            go.Scatter3d(x=curr_joints[:, 0], y=curr_joints[:, 1], z=curr_joints[:, 2], mode="markers", marker=dict(size=6, color="#ff3838")),
            go.Scatter3d(x=[w_pt[0], t_pt[0]], y=[w_pt[1], t_pt[1]], z=[w_pt[2], t_pt[2]], mode="lines", line=dict(color="#ffdd59", width=8))
        ]
        for p1_idx, p2_idx in BONES:
            p1, p2 = pose_pts[p1_idx], pose_pts[p2_idx]
            frame_traces.append(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode="lines", line=dict(color="#ffffff", width=4)))

        anim_frames.append(go.Frame(data=frame_traces, name=f"f_{idx}"))

    fig.frames = anim_frames
    fig.update_layout(
        paper_bgcolor="#1e272e", plot_bgcolor="#1e272e", font=dict(color="#ffffff"),
        scene=dict(
            xaxis=dict(title="左右 (X)", backgroundcolor="#2d3436", gridcolor="#485460"),
            yaxis=dict(title="前後 (Y)", backgroundcolor="#2d3436", gridcolor="#485460"),
            zaxis=dict(title="高度 (Z)", backgroundcolor="#2d3436", gridcolor="#485460"),
            aspectmode="data", camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30), height=600,
        updatemenus=[dict(type="buttons", showactive=False, y=0.05, x=0.05, buttons=[
            dict(label="▶ 播放 3D 揮棒", method="animate", args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
            dict(label="⏸ 暫停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
        ])]
    )
    return fig


# ─── Gemini AI 診斷生成 (穩定防錯版) ───────────────────────────────────
def generate_gemini_report(metrics_data) -> BaseballAnalysisReport:
    if not HAS_GENAI:
        raise RuntimeError("系統未能正確載入 google-genai 套件，請確認 requirements.txt。")

    # 1. 從 Streamlit Secrets 或環境變數讀取 API Key
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

    if api_key:
        api_key = str(api_key).strip()

    if not api_key:
        raise ValueError("未偵測到 GEMINI_API_KEY！請至 Streamlit Cloud 的 Settings -> Secrets 填入有效的 API Key。")

    # 2. 明確初始化 client 變數（避免 UnboundLocalError）
    try:
        client = genai.Client(api_key=api_key)
    except Exception as init_err:
        raise RuntimeError(f"Gemini Client 初始化失敗：{init_err}")

    system_instruction = """
  你是一位 MLB 級別的棒球生物力學專家與教練。
  請根據傳入的 3D 關節數據（最大 X-Factor 角度、手腕峰值移動速度、前膝角度），
  進行深度的運動機制診斷，指出潛在的能量流失瑕疵，並提供結構化的建議報告。
  """

    prompt_payload = {
        "max_x_factor_degrees": metrics_data["max_x_factor"],
        "max_wrist_speed_mph": metrics_data["max_wrist_speed_mph"],
        "avg_knee_angle_degrees": metrics_data["avg_knee_angle"],
        "total_analyzed_frames": len(metrics_data["frame_indices"])
    }

    prompt = f"請分析以下棒球運動生物力學數據並生成報告：\n```json\n{json.dumps(prompt_payload, indent=2)}\n```"

    # 3. 呼叫 Gemini 3.8 Flash 模型
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=BaseballAnalysisReport,
        ),
    )
    return BaseballAnalysisReport.model_validate_json(response.text)


# ─── Streamlit 主 UI 介面 ───────────────────────────────────────────
st.title("⚾ 棒球 3D 生物力學與 AI 診斷系統")
st.markdown("結合 MediaPipe 3D 世界座標、 Kinematic 曲線分析、單次/多次 3D 揮棒建模與 Gemini AI 診斷引擎。")

st.sidebar.header("⚙️ 分析參數設定")
tracked_joint_name = st.sidebar.selectbox("選擇追蹤關節", options=["右手腕 (16)", "左手腕 (15)", "右肘 (14)", "左肘 (13)"], index=0)
joint_map = {"右手腕 (16)": 16, "左手腕 (15)": 15, "右肘 (14)": 14, "左肘 (13)": 13}
tracked_joint = joint_map[tracked_joint_name]

sample_step = st.sidebar.slider("抽格採樣間隔 (Sample Step)", 1, 5, 1)

uploaded_file = st.sidebar.file_uploader("上傳影片 (MP4 / MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    suffix = os.path.splitext(uploaded_file.name)[1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📹 原始影片")
        st.video(video_bytes)

    with col2:
        try:
            records, fps, total_frames = analyze_video(video_bytes, suffix, sample_step)
            if records:
                metrics_data, poses_rel = compute_biomechanics_timeline(records, fps, tracked_joint)

                # 切割揮棒動作選單
                swings = segment_swings(metrics_data, fps)
                st.subheader("🧊 3D 姿勢與揮棒軌跡建模")

                selected_swing_name = st.selectbox(
                    "🎯 選擇 3D 建模視角 (支援單次揮棒與全景多次揮棒)：",
                    options=[s["name"] for s in swings]
                )
                selected_swing = next(s for s in swings if s["name"] == selected_swing_name)

                # 裁切出對應的揮棒片段進行 3D 繪製
                sub_records = records[selected_swing["start_idx"]: selected_swing["end_idx"] + 1]
                sub_poses = poses_rel[selected_swing["start_idx"]: selected_swing["end_idx"] + 1]

                fig_3d = make_3d_figure(sub_records, sub_poses, tracked_joint)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)

                st.subheader("📈 生物力學特徵曲線 (X-Factor & Speed)")
                fig_metrics = make_metrics_chart(metrics_data)
                st.plotly_chart(fig_metrics, use_container_width=True)

                st.subheader("🤖 Gemini 生物力學 AI 診斷報告")
                if "gemini_report" not in st.session_state:
                    st.session_state.gemini_report = None

                if st.button("🚀 呼叫 Gemini AI 生成診斷報告", type="primary"):
                    with st.spinner("Gemini API 分析動作機制中..."):
                        try:
                            st.session_state.gemini_report = generate_gemini_report(metrics_data)
                        except Exception as err:
                            st.error(f"Gemini 診斷失敗：{err}")

                if st.session_state.gemini_report:
                    report = st.session_state.gemini_report
                    st.metric("綜合評分", f"{report.overall_score} / 100")
                    st.info(f"**總結評語**：{report.summary}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**扭力分析 (X-Factor)**：{report.peak_x_factor_eval}")
                    with c2:
                        st.markdown(f"**速度分析 (Speed)**：{report.wrist_speed_eval}")

                    st.markdown("#### ⚠️ 檢測到的動作瑕疵")
                    for issue in report.detected_issues:
                        st.error(f"**[{issue.severity}風險] {issue.issue_name}**：{issue.analysis}")

                    st.markdown("#### 🏋️ 推薦矯正菜單 (Drills)")
                    for drill in report.recommended_drills:
                        st.success(f"**{drill.drill_name}** ({drill.objective})\n\n說明：{drill.execution}")
            else:
                st.warning("影片中未能辨識出清晰的人體骨架。")
        except Exception as e:
            st.error(f"分析過程發生錯誤：{e}")
else:
    st.info("👈 請於左側邊欄上傳影片檔以開始分析。")
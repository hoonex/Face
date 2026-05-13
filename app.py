import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

# MediaPipe 얼굴 인식 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def analyze_face(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # 이미지를 RGB로 변환 (MediaPipe는 RGB를 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, "얼굴을 찾을 수 없습니다. 정면 사진을 다시 업로드해주세요."

        # 이미지 크기
        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # 주요 랜드마크 좌표 추출 (픽셀 단위)
        def get_pt(index):
            return (int(landmarks[index].x * w), int(landmarks[index].y * h))

        # 1. 얼굴 길이 (이마 끝 - 턱 끝)
        top_of_head = get_pt(10)
        chin = get_pt(152)
        face_length = calculate_distance(top_of_head, chin)

        # 2. 얼굴 너비 (양쪽 광대)
        left_cheek = get_pt(234)
        right_cheek = get_pt(454)
        face_width = calculate_distance(left_cheek, right_cheek)

        # 3. 이마 너비
        left_forehead = get_pt(162)
        right_forehead = get_pt(389)
        forehead_width = calculate_distance(left_forehead, right_forehead)

        # 4. 턱 너비
        left_jaw = get_pt(132)
        right_jaw = get_pt(361)
        jaw_width = calculate_distance(left_jaw, right_jaw)

        # 비율 계산
        length_width_ratio = face_length / face_width
        jaw_width_ratio = jaw_width / face_width
        forehead_jaw_ratio = forehead_width / jaw_width

        # 얼굴형 판별 알고리즘
        shape_name = ""
        analysis_text = ""
        tips = ""

        if length_width_ratio > 1.4:
            shape_name = "긴 얼굴형 (Oblong)"
            analysis_text = f"가로 너비에 비해 세로 길이가 긴 편입니다. (세로/가로 비율: {length_width_ratio:.2f}). 이목구비가 뚜렷해 보이고 성숙한 인상을 줍니다."
            tips = "앞머리를 내려 세로 면적을 줄이거나, 옆머리에 볼륨을 주어 가로폭을 시각적으로 넓히는 스타일링이 잘 어울립니다."
        elif jaw_width_ratio > 0.85:
            shape_name = "각진 얼굴형 (Square)"
            analysis_text = f"이마, 광대, 턱의 너비가 비슷하며 턱선이 뚜렷합니다. (턱/광대 비율: {jaw_width_ratio:.2f}). 강인하고 신뢰감 있는 인상을 줍니다."
            tips = "직선적인 느낌을 부드럽게 완화하기 위해 둥근 프레임의 안경이나, 자연스러운 웨이브 헤어스타일을 추천합니다."
        elif forehead_jaw_ratio > 1.2:
            shape_name = "하트형 (Heart)"
            analysis_text = f"이마가 넓고 턱으로 갈수록 좁아지는 V라인을 가졌습니다. 이마와 턱의 대비가 뚜렷합니다. 사랑스럽고 날렵한 인상을 줍니다."
            tips = "넓은 이마를 자연스럽게 커버하는 시스루 뱅 앞머리나, 턱 주변에 볼륨을 주는 단발 스타일이 매력을 극대화합니다."
        elif length_width_ratio > 1.25:
            shape_name = "계란형 (Oval)"
            analysis_text = f"얼굴의 세로가 가로보다 살짝 길고, 전체적으로 굴곡 없이 부드러운 윤곽을 가졌습니다. (이상적인 비율에 가깝습니다.)"
            tips = "비율이 좋아 어떤 헤어스타일이나 안경, 모자도 무난하게 잘 소화할 수 있는 축복받은 얼굴형입니다."
        else:
            shape_name = "둥근 얼굴형 (Round)"
            analysis_text = f"가로와 세로 길이가 비슷하며, 볼살이 약간 있고 턱선이 부드럽습니다. (세로/가로 비율: {length_width_ratio:.2f}). 동안이며 친근한 인상을 줍니다."
            tips = "시선을 위아래로 분산시키기 위해 가르마를 타서 이마를 드러내거나, 얼굴선을 가려주는 레이어드 컷이 좋습니다."

        # 시각화 (이미지에 선과 점 그리기)
        annotated_image = image.copy()
        pts = [top_of_head, chin, left_cheek, right_cheek, left_forehead, right_forehead, left_jaw, right_jaw]
        
        # 핵심 포인트 그리기
        for pt in pts:
            cv2.circle(annotated_image, pt, 5, (0, 255, 0), -1)
        
        # 측정 선 그리기 (얼굴 길이, 너비)
        cv2.line(annotated_image, top_of_head, chin, (255, 0, 0), 2)
        cv2.line(annotated_image, left_cheek, right_cheek, (0, 0, 255), 2)

        return annotated_image, {
            "shape": shape_name,
            "analysis": analysis_text,
            "tips": tips,
            "metrics": {
                "length_width_ratio": length_width_ratio,
                "jaw_width_ratio": jaw_width_ratio
            }
        }

# Streamlit UI 구성
st.set_page_config(page_title="AI 얼굴형 정밀 분석기", page_icon="🧑‍🦲", layout="wide")

st.title("🤖 AI 얼굴형 정밀 분석 시스템")
st.markdown("최신 컴퓨터 비전 기술(MediaPipe)을 활용해 얼굴의 468개 랜드마크를 추출하여 비율을 정밀 계산합니다.")

uploaded_file = st.file_uploader("정면 사진을 업로드해주세요 (jpg, png, jpeg)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 이미지를 OpenCV 형식으로 읽기
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # RGBA인 경우 RGB로 변환
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    st.info("AI가 얼굴 특징점을 분석하고 있습니다...")
    
    result_img, analysis_data = analyze_face(image)
    
    if result_img is None:
        st.error(analysis_data)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 특징점 인식 결과")
            # OpenCV는 BGR, Streamlit은 RGB를 사용하므로 변환
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        with col2:
            st.subheader("📊 상세 분석 리포트")
            
            st.success(f"**판별된 얼굴형: {analysis_data['shape']}**")
            
            st.markdown("### 🔍 형태 분석")
            st.write(analysis_data['analysis'])
            
            st.markdown("### 💡 추천 스타일링")
            st.info(analysis_data['tips'])
            
            st.markdown("### 📏 정밀 측정 수치")
            st.metric(label="세로/가로 비율 (1.2~1.4가 일반적)", value=f"{analysis_data['metrics']['length_width_ratio']:.2f}")
            st.metric(label="턱 너비 비율 (광대 대비)", value=f"{analysis_data['metrics']['jaw_width_ratio']:.2f}")
            
            st.caption("※ 분석 결과는 사진의 각도와 조명에 따라 달라질 수 있습니다. 최대한 정면을 바라보고 이마와 턱이 다 보이는 사진일수록 정확합니다.")

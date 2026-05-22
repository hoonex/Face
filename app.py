import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
import os

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def extract_face_vector(image_rgb):
    """이미지에서 얼굴의 8가지 정밀 비율 벡터를 추출하는 핵심 함수"""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, None

        h, w, _ = image_rgb.shape
        landmarks = results.multi_face_landmarks[0].landmark

        def get_pt(index):
            return (int(landmarks[index].x * w), int(landmarks[index].y * h))

        # 랜드마크 추출
        top_of_head, chin = get_pt(10), get_pt(152)
        left_cheek, right_cheek = get_pt(234), get_pt(454)
        left_jaw, right_jaw = get_pt(132), get_pt(361)
        left_forehead, right_forehead = get_pt(162), get_pt(389)
        glabella, nose_base = get_pt(9), get_pt(2)
        left_eye_outer, left_eye_inner = get_pt(33), get_pt(133)
        right_eye_inner, right_eye_outer = get_pt(362), get_pt(263)
        nose_left, nose_right = get_pt(129), get_pt(358)
        lip_top, lip_bottom = get_pt(13), get_pt(14)
        lip_left, lip_right = get_pt(61), get_pt(291)

        # 물리적 거리 계산
        face_length = calculate_distance(top_of_head, chin)
        face_width = calculate_distance(left_cheek, right_cheek)
        jaw_width = calculate_distance(left_jaw, right_jaw)
        forehead_width = calculate_distance(left_forehead, right_forehead)
        upper_face = calculate_distance(top_of_head, glabella)
        mid_face = calculate_distance(glabella, nose_base)
        lower_face = calculate_distance(nose_base, chin)
        left_eye_width = calculate_distance(left_eye_outer, left_eye_inner)
        right_eye_width = calculate_distance(right_eye_inner, right_eye_outer)
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        interocular_dist = calculate_distance(left_eye_inner, right_eye_inner)
        nose_width = calculate_distance(nose_left, nose_right)
        lip_height = calculate_distance(lip_top, lip_bottom)
        lip_width = calculate_distance(lip_left, lip_right)

        # 비율 계산 (분모가 0이 되는 것 방지)
        safe_div = lambda n, d: n / d if d else 1.0
        
        vector = [
            safe_div(face_length, face_width),    # 세로/가로 비
            safe_div(jaw_width, face_width),      # 턱/광대 비
            safe_div(forehead_width, jaw_width),  # 이마/턱 비
            safe_div(upper_face, mid_face),       # 상안부 비
            safe_div(lower_face, mid_face),       # 하안부 비
            safe_div(interocular_dist, avg_eye_width), # 미간/눈 비
            safe_div(nose_width, interocular_dist),    # 코/미간 비
            safe_div(lip_height, lip_width)            # 입술 두께 비
        ]
        
        return vector, {
            "top_of_head": top_of_head, "chin": chin, 
            "left_cheek": left_cheek, "right_cheek": right_cheek,
            "left_eye_inner": left_eye_inner, "right_eye_inner": right_eye_inner,
            "nose_left": nose_left, "nose_right": nose_right
        }

@st.cache_data
def build_dynamic_db(folder_path="celeb_images"):
    """
    폴더 내의 이미지를 AI가 직접 스캔하여 실시간 매칭 DB를 구축합니다.
    Streamlit 캐시를 사용하여 최초 1회만 스캔하고 속도를 최적화합니다.
    """
    db = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return db
        
    valid_ext = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_ext):
            filepath = os.path.join(folder_path, filename)
            # 한글 경로명 지원을 위해 np.fromfile 사용
            img_array = np.fromfile(filepath, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vector, _ = extract_face_vector(img_rgb)
                if vector:
                    name = os.path.splitext(filename)[0] # 파일명(확장자 제외)을 이름으로 사용
                    db[name] = vector
    return db

def find_closest_match(user_vector, celeb_db):
    if not celeb_db:
        return "데이터 없음", 0.0
        
    best_match = None
    min_distance = float('inf')
    
    for name, celeb_vector in celeb_db.items():
        dist = math.sqrt(sum((u - c) ** 2 for u, c in zip(user_vector, celeb_vector)))
        if dist < min_distance:
            min_distance = dist
            best_match = name
            
    # 유클리디안 거리를 100점 만점 유사도로 변환 (경험적 수치)
    similarity = max(0.0, 100.0 - (min_distance * 70))
    return best_match, similarity

# Streamlit UI 구성
st.set_page_config(page_title="AI 관상 & 무제한 매칭 시스템", page_icon="♾️", layout="wide")

st.title("♾️ AI 안면 정밀 분석 및 자율 학습 매칭")
st.markdown("하드코딩된 데이터는 없습니다. `celeb_images` 폴더에 원하는 사진을 넣으면 AI가 알아서 특징을 추출하고 학습하여 당신과 가장 닮은 비율을 찾아냅니다.")

# --- 동적 DB 로드 ---
CELEB_FOLDER = "celeb_images"
with st.spinner("AI가 폴더 내의 이미지 데이터를 학습하고 있습니다... 🧠"):
    celeb_db = build_dynamic_db(CELEB_FOLDER)

if not celeb_db:
    st.warning(f"⚠️ `{CELEB_FOLDER}` 폴더가 비어있거나 생성되었습니다. 프로젝트 폴더 안에 `{CELEB_FOLDER}` 폴더를 만들고, 비교하고 싶은 인물들의 정면 사진을 넣어주세요! (파일명 예시: 카리나.jpg)")

uploaded_file = st.file_uploader("정확한 분석을 위해 앞머리를 넘긴 정면 사진을 올려주세요.", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    with st.spinner("사용자의 8차원 얼굴 특징점 벡터를 추출 중... ⚙️"):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        user_vector, pts = extract_face_vector(image_rgb)
    
    if user_vector is None:
        st.error("얼굴을 찾을 수 없습니다. 마스크나 안경이 없는 정면 사진인지 확인해주세요.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📡 안면 랜드마크 스캐닝")
            annotated_image = image.copy()
            
            # 시각화 포인트 그리기
            for key, pt in pts.items():
                cv2.circle(annotated_image, pt, 6, (255, 255, 255), -1)
                cv2.circle(annotated_image, pt, 3, (0, 0, 0), -1)
                
            cv2.line(annotated_image, pts["top_of_head"], pts["chin"], (255, 200, 0), 2)
            cv2.line(annotated_image, pts["left_cheek"], pts["right_cheek"], (0, 255, 255), 2)
            cv2.line(annotated_image, pts["left_eye_inner"], pts["right_eye_inner"], (0, 255, 0), 2)
            cv2.line(annotated_image, pts["nose_left"], pts["nose_right"], (255, 0, 0), 2)
            
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        with col2:
            st.subheader("🎯 동적 매칭 알고리즘 결과")
            
            match_name, match_score = find_closest_match(user_vector, celeb_db)
            
            if celeb_db:
                st.success(f"### 가장 비슷한 비율의 인물: **{match_name}**")
                st.progress(int(match_score))
                st.caption(f"수학적 형태 유사도: {match_score:.1f}%")
            else:
                st.error("비교할 대상이 없습니다. 사진을 폴더에 추가해주세요.")
            
            st.markdown("---")
            st.markdown("### 🔬 추출된 핵심 비율 데이터")
            
            st.metric(label="얼굴 세로/가로 비율", value=f"{user_vector[0]:.2f}")
            
            c1, c2 = st.columns(2)
            c1.metric(label="상안부 비율 (중안부 1 기준)", value=f"{user_vector[3]:.2f}")
            c2.metric(label="하안부 비율 (중안부 1 기준)", value=f"{user_vector[4]:.2f}")
            
            c3, c4 = st.columns(2)
            c3.metric(label="미간 너비 (눈 1 기준)", value=f"{user_vector[5]:.2f}")
            c4.metric(label="코볼 너비 (미간 1 기준)", value=f"{user_vector[6]:.2f}")

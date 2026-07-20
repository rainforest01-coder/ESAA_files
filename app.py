import streamlit as st
import numpy as np
import pandas as pd
import re
import google.generativeai as genai
from googleapiclient.discovery import build
import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# 제미나이 API 설정 초기화
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# ⚙️ [백엔드 기능 코드 - 유튜브 분석용]
# ==========================================

# [1단계] URL 파싱 (일반 영상 및 쇼츠 Video ID 모두 추출 가능)
def extract_video_id(url):
    # 정규표현식에 shorts\/ 패턴을 추가하여 쇼츠 주소도 완벽히 대응합니다.
    regex = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|shorts\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    match = re.search(regex, url)
    return match.group(1) if match else None

# [2단계] YouTube Data API 호출 (제목, 설명, 댓글 20개)
def get_youtube_metadata(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()
    
    if not video_response['items']:
        return None, None, []
        
    title = video_response['items'][0]['snippet']['title']
    description = video_response['items'][0]['snippet']['description']
    
    comments = []
    try:
        comment_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=20,
            order="relevance"
        ).execute()
        
        for item in comment_response['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment_text)
    except Exception:
        comments.append("댓글 기능을 사용할 수 없거나 댓글이 없습니다.")
        
    return title, description, comments

# [3단계] 제미나이 AI 분석 및 가시성 정제 (정보 취약층 타겟 기조 강화)
def analyze_with_gemini(youtube_data):

    model = genai.GenerativeModel("models/gemini-3.1-flash-lite")

    prompt = f"""
당신은 미디어 리터러시 전문가입니다.

다음 유튜브 콘텐츠(제목, 설명, 자막, 댓글)를 분석하여
콘텐츠의 감정 자극 및 허위정보 위험 가능성을 평가하세요.

[분석 우선순위]
1. 영상 제목
2. 영상 설명
3. 자막
4. 댓글(참고용)

제목, 설명, 자막을 중심으로 콘텐츠를 분석하세요.

{youtube_data}

------------------------------------
다음 요소만으로는 선동성이 높다고 판단하지 마세요.

- 일반적인 클릭 유도 문구
- 건강 정보 콘텐츠의 일반적인 제목
- 경제 뉴스의 일반적인 제목
- 제품 홍보 문구
- 언론 기사에서 흔히 사용하는 표현

------------------------------------
[점수 산정 원칙]

다음 요소가 하나 이상 확인될 경우에는 일반적인 정보성 콘텐츠보다 높은 점수를 부여하세요.

- 허위정보로 오인될 가능성이 높은 표현
- 공포나 분노를 과도하게 자극하는 표현
- 근거 없이 사실처럼 단정하는 주장
- 특정 집단에 대한 적대감을 유도하는 표현
- 사회적 불안을 의도적으로 확대하는 표현
- 음모론적 또는 검증이 어려운 주장
- 사실 확인을 어렵게 만드는 과장 표현

특히 여러 요소가 동시에 발견되는 경우에는
콘텐츠가 허위정보를 확산하거나 잘못된 판단을 유도할 위험이 높다고 보고
60점 이상을 부여하세요.

허위정보 위험 요소가 매우 강하거나,
사실과 의견을 구분하지 않고 단정적으로 전달하는 경우에는
80점 이상의 점수를 부여하세요.

반대로 정보 전달 목적이 명확하고
객관적인 근거를 중심으로 설명하는 콘텐츠는
20점 이하를 부여하세요.

------------------------------------
[근거 작성 규칙]

- 실제 확인된 내용만 작성하세요.
- 추측은 작성하지 마세요.
- 근거는 최대 3개까지만 작성하세요.
- 위험 요소가 거의 없다면 근거는 1개 이하만 작성하세요.

------------------------------------
출력 형식

점수: 숫자

근거:
- ...
- ...
- ...
"""

    response = model.generate_content(prompt)

    raw_text = response.text
    
    beautiful_result = []
    
    # 1. 점수 파싱 및 위험도 이모지 추가
    score_match = re.search(r"점수:\s*(\d+)", raw_text)
    if score_match:
        score = int(score_match.group(1))
        if score >= 70:
            level = "🚨 위험"
            action_guide = (
                "💡 **미디어 수용 가이드**\n"
                "  - 본 콘텐츠는 공포심이나 분노 등 자극적 감정을 유도하는 표현의 빈도가 높게 분석되었습니다.\n"
                "  - 감정적 판단을 유도하는 콘텐츠의 특성상, 정보의 무분별한 확산을 방지하기 위해 신중한 시청 및 공유 조절이 권장됩니다."
            )
        elif score >= 40:
            level = "⚠️ 주의"
            action_guide = (
                "💡 **미디어 수용 가이드**\n"
                "  - 본 콘텐츠는 객관적 사실과 주관적 주장이 혼재되어 있어 시청자의 편향을 유도할 가능성이 있습니다.\n"
                "  - 균형 잡힌 시각을 확보하기 위해 공신력 있는 기관이나 언론사의 보도 내용을 통한 추가적인 교차 검증이 필요합니다."
            )
        else:
            level = "✅ 안전"
            action_guide = (
                "💡 **미디어 수용 가이드**\n"
                "  - 본 콘텐츠는 과도한 감정 자극이나 왜곡적 표현의 비중이 낮아 상대적으로 객관성을 유지하고 있는 것으로 분석됩니다.\n"
                "  - 다만 온라인 콘텐츠의 특성을 고려하여, 세부적인 사실관계는 참고용으로 수용하는 것이 바람직합니다."
            )
        beautiful_result.append(
            f"### 📊 콘텐츠 선동성 분석 점수\n\n"
            f"#### **{score} / 100 점** {level}\n")
    else:
        beautiful_result.append("📊  콘텐츠 선동성 분석 점수  【 측정 불가 】\n")
        
    beautiful_result.append("---------------------------------------------------------------\n")
    beautiful_result.append("📝 **AI 분석 근거**\n")
    
    # 2. 이유(근거) 파싱 및 기호 변경
    lines = raw_text.split('\n')
    for line in lines:

        line = line.strip()

        if line.startswith("-"):

            clean = line[1:].strip()

            if clean == "해당 없음":
                continue

            beautiful_result.append(f"• {clean}")
            
            beautiful_result.append("\n")
    
    # 🌟 [신규 추가] 위험도별 맞춤 가이드 코멘트 출력 파트
    if action_guide:
        beautiful_result.append("---------------------------------------------------------------\n")
        beautiful_result.append(f"{action_guide}\n")
        
    beautiful_result.append("----------------------------------------------------")
    beautiful_result.append(
        "※ 본 분석은 제목, 설명 및 댓글을 기반으로 생성된 AI 참고 결과이며, 콘텐츠의 사실 여부를 판정하는 것은 아닙니다."
    )
    
    return "\n".join(beautiful_result)


# ==========================================
# 🎨 [Streamlit 프론트엔드 통합 메인 UI]
# ==========================================

# 1. 전체 페이지 공통 설정
st.set_page_config(
    page_title="허위정보 예방 서비스",
    page_icon="📰",
    layout="centered"
)

st.title("허위정보 예방 서비스")
st.markdown("""
소셜미디어 이용자의 뉴스 이용 특성을 진단하고,
유튜브 콘텐츠의 감정 자극 표현과 선동 가능성을 분석하는 서비스입니다.
""")
# 📌 탭 메뉴 생성 (1탭: 자가진단, 2탭: 유튜브 검증)
tab1, tab2 = st.tabs(["📋 허위정보 취약성 자가진단", "🚀 유튜브 실시간 위험성 검증"])

# ------------------------------------------------------------------------------
# 탭 1: 허위정보 취약성 자가진단
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("📋 평소 뉴스 이용 습관에 대해 응답해 주세요.")
    # 1. 연령 (SQ2)
    q1_age = st.number_input("1. 귀하의 연령은 어떻게 되십니까? (만 나이 입력)", min_value=1, max_value=120, value=30,key="diag_age")

    # 2. 뉴스 출처 확인 빈도 (G12)
    q2_g12 = st.selectbox(
        "2. 귀하께서는 소셜미디어에서 접하는 뉴스를 보도한 언론사명을 얼마나 자주 확인하십니까?",
        options=["① 전혀 확인하지 않는다", "② 거의 확인하지 않는다", "③ 가끔 확인한다", "④ 자주 확인한다", "⑤ 항상 확인한다"],
        index=2
    )

    # 3. 뉴스 이용 방식 (G3) - 원본 설문지 보기 100% 완벽 복원
    st.markdown("**3. 귀하께서는 어떤 방식으로 뉴스/시사정보를 이용하십니까? 다음에 제시된 보기 중에서 많이 이용하시는 방식을 모두 선택해 주십시오.**")
    g3_1 = st.checkbox("① 신문사/방송사 공식 채널/계정의 뉴스를 이용")
    g3_2 = st.checkbox("② 시사채널(개인, 단체) 채널/계정의 뉴스를 이용")
    g3_3 = st.checkbox("③ 알고리즘이 추천하는 뉴스를 이용")
    g3_4 = st.checkbox("④ 친구나 지인이 공유 또는 전달한 뉴스를 이용")
    g3_5 = st.checkbox("⑤ 관심 있는 이슈를 검색하여 뉴스를 이용")
    g3_6 = st.checkbox("⑥ 제목이나 썸네일이 눈에 띄는 뉴스를 이용")  # EDA 최상위 위험 경로 (코드 6번)
    g3_7 = st.checkbox("⑦ 기타")

    # 4. 뉴스 출처 인지 정도 (G11)
    q4_g11 = st.selectbox(
        "4. 귀하께서는 소셜미디어에서 접한 뉴스가 어느 언론사에서 작성/제공한 뉴스인지 어느 정도 알고 계십니까?",
        options=["① 전혀 알지 못한다", "② 알지 못하는 편이다", "③ 보통이다", "④ 아는 편이다", "⑤ 항상 알고 있다"],
        index=2
    )

    # 5. 뉴스 공유 행동 빈도 (G4_5)
    q5_g4_5 = st.selectbox(
        "5. 귀하께서는 소셜미디어에서 뉴스/시사정보를 이용할 때 [5. 뉴스를 다른 곳으로 공유하기] 행위를 얼마나 자주 하십니까?",
        options=["① 전혀 하지 않는다", "② 거의 하지 않는다", "③ 보통이다", "④ 자주 한다", "⑤ 매우 자주 한다"],
        index=2
    )

    # 6. 타인/지인 신뢰도 (점수 보정용 독립 문항)
    q6_score = st.radio(
        "6. 귀하께서는 메신저나 SNS를 통해 지인 또는 신뢰하는 사람이 공유해 준 뉴스일 경우, 별도의 사실 확인 절차 없이 신뢰하는 편이십니까?",
        options=["예, 대체로 신뢰하고 받아들이는 편이다", "아니오, 지인이 공유했더라도 내용의 사실 여부는 의심해보는 편이다"]
    )

    # 7. 교차 검증 습관 (점수 보정용 독립 문항)
    q7_score = st.selectbox(
        "7. 귀하께서는 소셜미디어에서 접한 뉴스를 타인에게 공유하기 전, 해당 내용이 사실인지 포털 사이트 등에 검색해 보는 등 교차 검증을 수행하십니까?",
        options=["항상 교차 검증을 수행한다", "자주 수행하는 편이다", "가끔 생각나면 수행한다", "거의 수행하지 않는다", "단 한 번도 해본 적 없다"],
        index=3
    )

    st.markdown("---")

    if st.button("📊 진단 결과 확인", use_container_width=True):
       # 변경된 설문 보기에 맞춤 매핑 및 위험 요인 추출
        g12_map = {"① 전혀 확인하지 않는다": 1, "② 거의 확인하지 않는다": 2, "③ 가끔 확인한다": 3, "④ 자주 확인한다": 4, "⑤ 항상 확인한다"}
        g12_val = g12_map[q2_g12]
        target_value = 1 if g12_val in [1, 2] else 0

        # 3번 문항 체크박스 중 '⑥ 제목이나 썸네일이 눈에 띄는 뉴스를 이용' 선택 여부 확인
        g3_thumbnail_flag = 1 if g3_6 else 0
        
        g11_map = {"① 전혀 알지 못한다": 1, "② 알지 못하는 편이다": 2, "③ 보통이다": 3, "④ 아는 편이다": 4, "⑤ 항상 알고 있다"}
        g11_val = g11_map[q4_g11]
        
        g4_5_map = {"① 전혀 하지 않는다": 1, "② 거의 하지 않는다": 2, "③ 보통이다": 3, "④ 자주 한다": 4, "⑤ 매우 자주 한다"}
        g4_5_val = g4_5_map[q5_g4_5]

        # 유형 판별 조건 작동 (취약층 전반으로 문구 톤 수정)
        if target_value == 0:
            user_type = "비판적 이용형"
            type_desc = "출처를 확인하고 다른 언론 보도와 비교하는 습관이 비교적 잘 형성되어 있습니다."
            alert_style = st.success
        else:
            if q1_age >= 60:
                user_type = "출처 확인 지원 필요형"
                type_desc= "뉴스를 이용할 때 출처를 확인하는 습관을 조금 더 기르면 정보의 신뢰성을 판단하는 데 도움이 됩니다."
                alert_style = st.warning
            elif g3_thumbnail_flag == 1:
                user_type = "썸네일 중심 이용형"
                type_desc = "제목이나 썸네일의 영향을 비교적 크게 받는 뉴스 이용 패턴을 보입니다. 눈길을 사로잡는 낚시성 헤드라인에 넘어가지 않게 주의가 필요합니다."                
                alert_style = st.error
            elif g11_val in [1, 2]:
                user_type = "출처 확인 미흡형"
                type_desc = "뉴스를 이용할 때 언론사와 출처를 함께 확인하는 습관을 권장합니다."
                alert_style = st.error
            elif g4_5_val in [4, 5]:
                user_type = "공유 빈도 높은 이용형"
                type_desc = "뉴스를 공유하기 전에 다른 언론사의 보도와 함께 확인하면 정보의 신뢰성을 높일 수 있습니다."
                alert_style = st.error
            else:
                user_type = "일반 이용형"
                type_desc = "평소 뉴스를 이용할 때 출처 확인과 교차 검증을 함께 하면 더욱 신뢰도 높은 정보 이용에 도움이 됩니다."
                alert_style = st.warning

        # 취약성 점수 산출
        risk_factor = (1.5 * g3_thumbnail_flag) + (1.0 * g4_5_val) + (3.0 if "대체로 믿습니다" in q6_score else 0.0)
        q7_map = {"항상 교차 검증을 해본다": 4, "자주 해보는 편이다": 3, "가끔 생각나면 한다": 2, "거의 하지 않는다": 1, "단 한 번도 해본 적 없다": 0}
        defense_factor = (1.2 * g11_val) + (1.5 * g12_val) + (1.5 * q7_map[q7_score])

        raw_score = risk_factor - defense_factor
        min_val, max_val = -19.5, 6.5
        scaled_score = ((raw_score - min_val) / (max_val - min_val)) * 100
        vulnerability_index = round(np.clip(scaled_score, 0, 100), 1)

        # 결과 출력 (직관성 보완 버전)
        st.subheader("📊 진단 결과")
        
                # 점수 대역별 상태 정의
        if vulnerability_index >= 70:
            status_label = "🚨 위험"
            improve_desc = (
                "뉴스를 공유하기 전 출처를 확인하고, 다른 언론사의 보도와 함께 비교해 보는 습관을 권장합니다."
            )

        elif vulnerability_index >= 40:
            status_label = "⚠️ 주의"
            improve_desc = (
                "제목이나 썸네일만으로 판단하기보다, 다른 언론사의 보도도 함께 확인하면 더욱 신뢰도 높은 정보 이용에 도움이 됩니다."
            )

        else:
            status_label = "✅ 안전"
            improve_desc = (
                "현재의 출처 확인 및 교차 검증 습관을 유지하면 신뢰도 높은 뉴스 이용에 도움이 됩니다."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="미디어 정보 확인 지수",
                value=f"{vulnerability_index}점",
                delta=status_label,
                delta_color="off"
            )

        with col2:
            st.metric(
                label="뉴스 이용 유형",
                value=user_type
            )

        st.progress(int(vulnerability_index))

        st.caption(
            "※ 본 결과는 뉴스 이용 습관과 출처 확인 행동을 바탕으로 산출된 참고용 분석 결과입니다."
        )

        st.info(f"**📌 이용 특성**\n\n{type_desc}")

        st.warning(f"**💡 개선하면 좋은 점**\n\n{improve_desc}")

# ------------------------------------------------------------------------------
# 탭 2: 유튜브 실시간 위험성 검증
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🚀 AI 기반 유튜브 콘텐츠 분석")

    st.markdown("""
    유튜브 영상 링크를 입력하면 영상 제목, 설명, 댓글, 자막(제공되는 경우)을 종합적으로 분석하여  
    감정 자극 표현, 과장된 주장, 출처 신뢰성 등을 바탕으로 콘텐츠의 **허위정보 위험 가능성**을 분석합니다.
    """)
    
    target_url = st.text_input(
        "🔗 분석할 유튜브 영상 링크를 입력하세요:",
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_url_input"
    )

    if st.button("🚀 실시간 AI 검증 시작", use_container_width=True, key="yt_analysis_btn"):
        if not target_url:
            st.warning("⚠️ 유튜브 링크를 입력해 주세요!")
        else:
            with st.spinner("🔄 유튜브 데이터를 수집하고 AI 분석 리포트를 생성하는 중입니다..."):
                try:
                    # [1단계] ID 추출
                    video_id = extract_video_id(target_url)
                    if not video_id:
                        st.error("❌ 올바른 형식의 유튜브 URL이 아닙니다. 다시 확인해 주세요.")
                    else:
                        # [2단계] 메타데이터 및 댓글 수집
                        title, description, comments = get_youtube_metadata(video_id)
                        if not title:
                            st.error("❌ 영상 정보를 가져올 수 없습니다. 비공개 영상이거나 삭제된 링크인지 확인해 주세요.")
                        else:
                            # 데이터 문자열 결합
                            combined_data = f"""
                            영상 제목: {title}
                            영상 설명: {description}
                            댓글 리스트: {" | ".join(comments)}
                            """
                            
                            # [3단계] 제미나이 분석
                            analysis_result = analyze_with_gemini(combined_data)
                            
                            # [4단계] 결과 화면 출력
                            st.success("🎉 실시간 AI 분석 성공!")
                            st.markdown(analysis_result)
                            
                except Exception as e:
                    st.error(f"❌ 분석 작동 중 오류가 발생했습니다: {e}")

# 전체 하단 크레딧 (정보 취약층으로 변경)
st.divider()
st.caption("© 2026 공모전 프로토타입 개발팀 - 디지털 정보 취약층 미디어 안전망 프로젝트")

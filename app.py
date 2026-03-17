import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import requests
from sklearn.ensemble import RandomForestClassifier

# RDKit 임포트
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

###-------- 머신러닝 모델 세팅 (캐싱) --------###
@st.cache_resource
def load_or_train_model():
    # 실제 현업에서는 joblib.load('model.pkl')을 사용하지만,
    # 프로토타입 시연을 위해 가상의 데이터로 랜덤포레스트 모델을 즉시 학습시킵니다.
    np.random.seed(42)
    # 가상 특성 데이터: [분자량(MolWt), 지용성(LogP)]
    X_train = np.random.rand(500, 2) * [600, 6] 
    
    # 지용성과 분자량이 특정 수치 이상일 때 내분비계 교란 위험이 높도록 가상의 정답지(Label) 생성
    y_train = []
    for mw, logp in X_train:
        if logp > 3.5 and 150 < mw < 400: y_train.append(2) # HIGH
        elif logp > 2.0: y_train.append(1) # MEDIUM
        else: y_train.append(0) # LOW
        
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

###-------- CAS 번호를 SMILES로 변환하는 함수 --------###
def get_smiles_from_cas(cas_no):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(cas_no)}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception:
        pass
    return None

###-------- 페이지 설정 --------###
st.set_page_config(page_title="EDC 예측 플랫폼", page_icon="🧬", layout="wide")

if 'menu_option' not in st.session_state:
    st.session_state['menu_option'] = "대시보드"
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'current_score' not in st.session_state:
    st.session_state['current_score'] = 0.0
if 'current_risk' not in st.session_state:
    st.session_state['current_risk'] = ""
if 'input_val' not in st.session_state:
    st.session_state['input_val'] = ""
if 'analyzed_smiles' not in st.session_state:
    st.session_state['analyzed_smiles'] = ""
if 'chem_props' not in st.session_state:
    st.session_state['chem_props'] = {}

def go_to_new_analysis():
    st.session_state['menu_option'] = "신규 분석"
    st.session_state['analysis_done'] = False

###-------- 스타일 설정 --------###
st.markdown("""
<style>
.main-title {font-size:26px; font-weight:bold; color:#1E3A8A; margin-bottom: 20px;}
.high {color:#EF4444; font-weight:bold; font-size:20px;}
.medium {color:#F59E0B; font-weight:bold; font-size:20px;}
.low {color:#22C55E; font-weight:bold; font-size:20px;}
.stProgress > div > div > div > div {background-color: #1E3A8A;}
</style>
""", unsafe_allow_html=True)

###-------- 사이드바 --------###
with st.sidebar:
    st.title("📊 EDC Screening")
    menu = st.radio("메뉴 선택", ["대시보드", "신규 분석"], key='menu_option')
    st.markdown("---")
    st.caption("Powered by Cheminformatics & ML")

###-------- 대시보드 --------###
if menu == "대시보드":
    st.session_state['analysis_done'] = False 
    st.markdown("<div class='main-title'>EDC Screening Dashboard</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 분석 물질", "1,024", "12 ↑")
    col2.metric("High Risk", "156", "3 ↑")
    col3.metric("Medium Risk", "312", "-")
    col4.metric("Low Risk", "556", "9 ↑")

    st.markdown("---")
    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        st.subheader("📈 주간 분석 현황")
        chart_data = pd.DataFrame(np.random.randint(5, 20, size=(7, 3)), columns=["High", "Medium", "Low"], index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        st.bar_chart(chart_data)
    with col_table:
        st.subheader("📋 최근 고위험 물질 목록")
        recent_data = pd.DataFrame({"CAS No.": ["80-05-7", "117-81-7", "25154-52-3"], "물질명": ["Bisphenol A", "DEHP", "Nonylphenol"], "위험도": ["HIGH", "HIGH", "HIGH"]})
        st.dataframe(recent_data, use_container_width=True, hide_index=True)
    st.button("새로운 물질 분석하기 🚀", on_click=go_to_new_analysis)

###-------- 신규 분석 --------###
elif menu == "신규 분석":
    st.markdown("<div class='main-title'>Step 1. 물질 입력</div>", unsafe_allow_html=True)

    input_type = st.radio("입력 방식", ["SMILES", "CAS Number"], horizontal=True)
    default_input = "CC(=O)OC1=CC=CC=C1C(=O)O" if input_type == "SMILES" else "68737-61-1"
    user_input = st.text_input("화학식 또는 번호를 입력하세요", value=default_input)

    if st.button("분석 실행 🔍"):
        with st.spinner("AI 모델이 화학적 특성을 추출하고 분석 중입니다..."):
            st.session_state['input_val'] = user_input.strip()
            smiles_to_analyze = st.session_state['input_val']
            
            # CAS Number 입력 시 PubChem API를 통해 SMILES 확보
            if input_type == "CAS Number":
                fetched_smiles = get_smiles_from_cas(smiles_to_analyze)
                if fetched_smiles:
                    smiles_to_analyze = fetched_smiles
                else:
                    st.error("해당 CAS 번호의 분자 구조(SMILES)를 DB에서 찾을 수 없습니다.")
                    st.stop()
            
            st.session_state['analyzed_smiles'] = smiles_to_analyze
            
            # RDKit으로 분자 특성(Features) 추출
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles_to_analyze)
                if mol:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    st.session_state['chem_props'] = {'MolWt': mw, 'LogP': logp}
                    
                    # ML 모델 예측 실행
                    rf_model = load_or_train_model()
                    probs = rf_model.predict_proba([[mw, logp]])[0]
                    
                    st.session_state['current_score'] = probs[2] # HIGH 클래스의 확률
                    
                    if probs[2] > 0.6: st.session_state['current_risk'] = "HIGH"
                    elif probs[1] > 0.4: st.session_state['current_risk'] = "MEDIUM"
                    else: st.session_state['current_risk'] = "LOW"
                    
                    st.session_state['analysis_done'] = True
                else:
                    st.error("유효하지 않은 구조식입니다. RDKit 분석에 실패했습니다.")
            else:
                st.error("머신러닝 특성 추출을 위해 서버에 RDKit 설치가 필요합니다.")

    if st.session_state['analysis_done']:
        st.markdown("---")
        st.markdown("<div class='main-title'>Step 2. AI 분석 결과 및 구조</div>", unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            risk = st.session_state['current_risk']
            score = st.session_state['current_score']
            
            if risk == "HIGH": st.markdown(f"<div class='high'>🔴 HIGH RISK (예측 확률: {score*100:.1f}%)</div>", unsafe_allow_html=True)
            elif risk == "MEDIUM": st.markdown(f"<div class='medium'>🟡 MEDIUM RISK (예측 확률: {score*100:.1f}%)</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div class='low'>🟢 LOW RISK (예측 확률: {score*100:.1f}%)</div>", unsafe_allow_html=True)

            st.write("")
            st.subheader("Estrogen Receptor Binding Probability")
            st.progress(int(score * 100))
            
            # 머신러닝에 사용된 Feature 공개
            st.info(f"**추출된 모델 입력 특성 (Features)**\n\n- 분자량 (MolWt): {st.session_state['chem_props']['MolWt']:.2f} g/mol\n- 지용성 (LogP): {st.session_state['chem_props']['LogP']:.2f}")
            
        with col_res2:
            encoded_id = urllib.parse.quote(st.session_state['analyzed_smiles'])
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_id}/PNG"
            st.image(img_url, caption="2D Structure", width=250)

        ###-------- 하위 정보 탭 --------###
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["작용기전 (AOP)", "규제 영향", "대체 물질 제안"])
        with tab1:
            st.write("**주요 타겟:** Estrogen Receptor (ER) / Androgen Receptor (AR)")
            st.write("**경로:** Receptor Binding → Gene Expression Alteration → Cellular Toxicity")
        with tab2:
            st.write("- **EU REACH:** SVHC 후보 목록 검토 필요")
        with tab3:
            st.write("유사 구조 대체 물질 검색 결과입니다.")

        ###-------- 보고서 다운로드 기능 --------###
        st.markdown("---")
        report_content = f"""========================================
[ 지능형 EDC Screening 분석 보고서 ]
========================================
1. 분석 물질 정보
- 입력값: {st.session_state['input_val']}
- SMILES 구조: {st.session_state['analyzed_smiles']}
- 분자량: {st.session_state['chem_props'].get('MolWt', 0):.2f}
- 지용성(LogP): {st.session_state['chem_props'].get('LogP', 0):.2f}

2. AI 예측 결과 (Random Forest Classifier)
- 종합 위험도: {st.session_state['current_risk']}
- ER 결합 예측 확률: {st.session_state['current_score']:.3f}
========================================"""
        st.download_button(label="📄 AI 분석 보고서 다운로드 (.txt)", data=report_content, file_name=f"EDC_ML_Report_{st.session_state['input_val']}.txt", mime="text/plain")

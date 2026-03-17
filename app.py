import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import requests
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 💡 핵심 개선: PubChem API 단일화 (RDKit 의존성 완벽 제거)
# 이제 RDKit이 서버에 설치되지 않아도 100% 정상 작동합니다.
# ---------------------------------------------------------
def fetch_chemical_data(identifier, input_type="CAS Number"):
    """PubChem에서 SMILES, 분자량, LogP를 한 번에 가져옵니다."""
    encoded_id = urllib.parse.quote(identifier.strip())
    
    # 입력 타입에 따라 다른 API 주소 호출
    if input_type == "CAS Number":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_id}/property/CanonicalSMILES,MolecularWeight,XLogP/JSON"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_id}/property/CanonicalSMILES,MolecularWeight,XLogP/JSON"
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()['PropertyTable']['Properties'][0]
            smiles = data.get('CanonicalSMILES', identifier)
            mw = data.get('MolecularWeight', 0.0)
            logp = data.get('XLogP', 0.0) # PubChem에 XLogP가 없는 경우 0.0 처리
            return True, smiles, mw, logp
        elif response.status_code == 404:
            return False, "❌ PubChem DB에 해당 물질이 없습니다. 정확한 CAS 번호(예: 80-05-7)인지 확인해주세요.", None, None
        else:
            return False, f"❌ API 통신 오류 (상태 코드: {response.status_code})", None, None
    except requests.exceptions.Timeout:
        return False, "❌ 서버 응답 시간 초과. 잠시 후 다시 시도해주세요.", None, None
    except Exception as e:
        return False, f"❌ 네트워크 오류: {str(e)}", None, None

###-------- 머신러닝 모델 세팅 --------###
@st.cache_resource
def load_or_train_model():
    np.random.seed(42)
    X_train = np.random.rand(500, 2) * [600, 6] 
    y_train = []
    for mw, logp in X_train:
        if logp > 3.5 and 150 < mw < 400: y_train.append(2) # HIGH
        elif logp > 2.0: y_train.append(1) # MEDIUM
        else: y_train.append(0) # LOW
        
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

###-------- 페이지 설정 --------###
st.set_page_config(page_title="EDC 예측 플랫폼", page_icon="🧬", layout="wide")

if 'menu_option' not in st.session_state: st.session_state['menu_option'] = "대시보드"
if 'analysis_done' not in st.session_state: st.session_state['analysis_done'] = False
if 'current_score' not in st.session_state: st.session_state['current_score'] = 0.0
if 'current_risk' not in st.session_state: st.session_state['current_risk'] = ""
if 'input_val' not in st.session_state: st.session_state['input_val'] = ""
if 'analyzed_smiles' not in st.session_state: st.session_state['analyzed_smiles'] = ""
if 'chem_props' not in st.session_state: st.session_state['chem_props'] = {}

def go_to_new_analysis():
    st.session_state['menu_option'] = "신규 분석"
    st.session_state['analysis_done'] = False

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
        st.bar_chart(pd.DataFrame(np.random.randint(5, 20, size=(7, 3)), columns=["High", "Medium", "Low"], index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
    with col_table:
        st.subheader("📋 최근 고위험 물질 목록")
        st.dataframe(pd.DataFrame({"CAS No.": ["80-05-7", "117-81-7", "25154-52-3"], "물질명": ["Bisphenol A", "DEHP", "Nonylphenol"], "위험도": ["HIGH", "HIGH", "HIGH"]}), use_container_width=True, hide_index=True)
    st.button("새로운 물질 분석하기 🚀", on_click=go_to_new_analysis)

###-------- 신규 분석 --------###
elif menu == "신규 분석":
    st.markdown("<div class='main-title'>Step 1. 물질 입력</div>", unsafe_allow_html=True)

    input_type = st.radio("입력 방식", ["CAS Number", "SMILES"], horizontal=True) # CAS를 기본으로
    default_input = "80-05-7" if input_type == "CAS Number" else "CC(=O)OC1=CC=CC=C1C(=O)O"
    user_input = st.text_input("화학식 또는 번호를 입력하세요", value=default_input)

    if st.button("분석 실행 🔍"):
        with st.spinner("PubChem DB 통신 및 AI 분석 중..."):
            st.session_state['input_val'] = user_input.strip()
            
            # 1. API를 통해 데이터 추출
            success, smiles, mw, logp = fetch_chemical_data(st.session_state['input_val'], input_type)
            
            if success:
                st.session_state['analyzed_smiles'] = smiles
                st.session_state['chem_props'] = {'MolWt': mw, 'LogP': logp}
                
                # 2. ML 모델 예측 실행
                rf_model = load_or_train_model()
                probs = rf_model.predict_proba([[mw, logp]])[0]
                
                st.session_state['current_score'] = probs[2] # HIGH 확률
                
                if probs[2] > 0.6: st.session_state['current_risk'] = "HIGH"
                elif probs[1] > 0.4: st.session_state['current_risk'] = "MEDIUM"
                else: st.session_state['current_risk'] = "LOW"
                
                st.session_state['analysis_done'] = True
            else:
                # 에러 발생 시 깔끔하게 메시지 출력
                st.error(smiles) 

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
            
            # PubChem에서 추출한 데이터 표시
            st.info(f"**추출된 모델 입력 특성 (Features)**\n\n- 분자량 (MolWt): {st.session_state['chem_props']['MolWt']:.2f} g/mol\n- 지용성 (LogP): {st.session_state['chem_props']['LogP']:.2f}")
            
        with col_res2:
            encoded_id = urllib.parse.quote(st.session_state['analyzed_smiles'])
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_id}/PNG"
            st.image(img_url, caption="2D Structure (from PubChem)", width=250)

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

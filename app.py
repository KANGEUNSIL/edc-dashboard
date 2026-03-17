import streamlit as st
import pandas as pd
import random
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

st.set_page_config(page_title="EDC 예측 플랫폼", page_icon="🧬", layout="wide")

if 'menu_option' not in st.session_state:
    st.session_state['menu_option'] = "대시보드"
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'current_score' not in st.session_state:
    st.session_state['current_score'] = 0.0
if 'current_risk' not in st.session_state:
    st.session_state['current_risk'] = ""
if 'input_smiles' not in st.session_state:
    st.session_state['input_smiles'] = ""

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

with st.sidebar:
    st.title("📊 EDC Screening")
    menu = st.radio("메뉴 선택", ["대시보드", "신규 분석"], key='menu_option')
    st.markdown("---")
    st.caption("Powered by Cheminformatics")

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
        chart_data = pd.DataFrame(
            np.random.randint(5, 20, size=(7, 3)),
            columns=["High", "Medium", "Low"],
            index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        )
        st.bar_chart(chart_data)

    with col_table:
        st.subheader("📋 최근 고위험 물질 목록")
        recent_data = pd.DataFrame({
            "CAS No.": ["80-05-7", "117-81-7", "25154-52-3", "118-74-1"],
            "물질명": ["Bisphenol A", "DEHP", "Nonylphenol", "Hexachlorobenzene"],
            "위험도": ["HIGH", "HIGH", "HIGH", "HIGH"],
            "신뢰도": ["98%", "95%", "92%", "89%"]
        })
        st.dataframe(recent_data, use_container_width=True, hide_index=True)

    st.button("새로운 물질 분석하기 🚀", on_click=go_to_new_analysis)

elif menu == "신규 분석":
    st.markdown("<div class='main-title'>Step 1. 물질 입력</div>", unsafe_allow_html=True)
    input_type = st.radio("입력 방식", ["SMILES", "CAS Number"], horizontal=True)
    default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" 
    user_input = st.text_input("화학식 또는 번호를 입력하세요", value=default_smiles)

    if st.button("분석 실행 🔍"):
        st.session_state['analysis_done'] = True
        st.session_state['input_smiles'] = user_input
        st.session_state['current_score'] = random.uniform(0, 1)
        score = st.session_state['current_score']
        if score > 0.7:
            st.session_state['current_risk'] = "HIGH"
        elif score > 0.4:
            st.session_state['current_risk'] = "MEDIUM"
        else:
            st.session_state['current_risk'] = "LOW"

    if st.session_state['analysis_done']:
        st.markdown("---")
        st.markdown("<div class='main-title'>Step 2. 분석 결과 및 구조</div>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            risk = st.session_state['current_risk']
            score = st.session_state['current_score']
            if risk == "HIGH":
                st.markdown(f"<div class='high'>🔴 HIGH RISK (Score: {score:.3f})</div>", unsafe_allow_html=True)
            elif risk == "MEDIUM":
                st.markdown(f"<div class='medium'>🟡 MEDIUM RISK (Score: {score:.3f})</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='low'>🟢 LOW RISK (Score: {score:.3f})</div>", unsafe_allow_html=True)
            st.write("")
            st.subheader("Estrogen Receptor Binding Probability")
            st.progress(int(score * 100))
            
        with col_res2:
            if RDKIT_AVAILABLE and input_type == "SMILES":
                try:
                    mol = Chem.MolFromSmiles(st.session_state['input_smiles'])
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption="2D Structure", width=250)
                    else:
                        st.warning("유효하지 않은 SMILES 코드입니다.")
                except Exception:
                    st.error("구조식 렌더링 중 오류가 발생했습니다.")
            elif not RDKIT_AVAILABLE:
                st.info("💡 구조식을 보려면 RDKit이 필요합니다.")

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["작용기전 (AOP)", "규제 영향", "대체 물질 제안"])
        
        with tab1:
            st.write("**주요 타겟:** Estrogen Receptor (ER) / Androgen Receptor (AR)")
            st.write("**경로:** Receptor Binding → Gene Expression Alteration → Cellular Toxicity")
            st.info("이 물질은 생체 내 호르몬 수용체와 결합하여 내분비계 교란을 일으킬 가능성이 높습니다.")
            
        with tab2:
            st.write("- **EU REACH:** SVHC (고위험성 우려 물질) 후보 목록 검토 필요")
            st.write("- **K-REACH:** 화학물질 등록 시 내분비계 교란 독성 시험자료 제출 요구 가능성 있음")
            
        with tab3:
            alt_data = pd.DataFrame({
                "추천 대체 물질 (SMILES)": ["CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccc(CC(=O)O)cc1"],
                "유사도 (Tanimoto)": ["85%", "78%"],
                "예상 위험도": ["LOW", "LOW"]
            })
            st.dataframe(alt_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("📄 PDF 보고서 다운로드"):
            st.success("보고서가 성공적으로 생성되었습니다! (Mock 다운로드 시작)")
            st.balloons()

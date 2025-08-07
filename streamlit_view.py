from typing import Any, Dict
import pandas as pd
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from agent import create_agent

# Custom callback handler for Streamlit streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.full_response = ""
        self.cursor = "▌"
        self.tool_output = ""
        self.is_tool_running = False
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.full_response += token
        self.placeholder.markdown(self.full_response + self.cursor)
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Run when LLM ends running."""
        self.placeholder.markdown(self.full_response)
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Run when tool starts running."""
        self.is_tool_running = True
        tool_name = serialized.get("name", "tool")
        self.full_response += f"\n\n*도구 실행 중: {tool_name}...*\n"
        self.placeholder.markdown(self.full_response + self.cursor)

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Run when tool ends running."""
        self.full_response = ""
        self.is_tool_running = False
        self.placeholder.markdown(self.full_response + self.cursor)
        
    def on_agent_action(self, action, **kwargs) -> Any:
        """Run on agent action."""
        pass
        
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Run on agent end."""
        pass

st.set_page_config(page_title="Azure Migration Agent", page_icon="🤖")
st.title("Azure Migration Agent")

# 서비스 정보 세션 상태 초기화
if "service_type" not in st.session_state:
    st.session_state.service_type = "VM 기반 서비스"
if "network_type" not in st.session_state:
    st.session_state.network_type = "공인망 연동"
if "service_abbr" not in st.session_state:
    st.session_state.service_abbr = ""
if "nas_usage" not in st.session_state:
    st.session_state.nas_usage = False

# 추가 정보 입력 섹션
st.sidebar.title("서비스 정보 입력")
service_type = st.sidebar.radio("서비스 유형", ["VM 기반 서비스", "AKS 서비스"], index=0 if st.session_state.service_type == "VM 기반 서비스" else 1)
network_type = st.sidebar.radio("네트워크 연동 유형", ["공인망 연동", "내부망 연동"], index=0 if st.session_state.network_type == "공인망 연동" else 1)
service_abbr = st.sidebar.text_input("서비스 약어를 입력하세요", value=st.session_state.service_abbr)
nas_usage = st.sidebar.checkbox("NAS 사용 여부", value=st.session_state.nas_usage)

# 세션 상태 업데이트
st.session_state.service_type = service_type
st.session_state.network_type = network_type
st.session_state.service_abbr = service_abbr
st.session_state.nas_usage = nas_usage

# 파일 업로드 위젯
uploaded_file = st.sidebar.file_uploader("Excel 또는 CSV 파일을 업로드하세요", type=["xlsx", "csv"])

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = None
if "user_notes" not in st.session_state:
    st.session_state.user_notes = ""
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "saved_configs" not in st.session_state:
    st.session_state.saved_configs = {}

# 파일 처리 함수
def process_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:  # Excel 파일
            df = pd.read_excel(file)
        
        # 여기서 데이터 처리 로직을 추가할 수 있습니다
        # 예: 간단한 통계 계산, 데이터 정리 등
        
        return df
    except Exception as e:
        st.sidebar.error(f"파일 처리 중 오류가 발생했습니다: {e}")
        return None


# 파일이 업로드되면 처리
if uploaded_file is not None:
    st.sidebar.info(f"파일 '{uploaded_file.name}'이(가) 업로드되었습니다.")
    
    # 파일 처리
    processed_df = process_file(uploaded_file)
    if processed_df is not None:
        st.session_state.processed_data = processed_df
        
        # 데이터 미리보기
        st.sidebar.subheader("데이터 미리보기")
        st.sidebar.dataframe(processed_df.head())
        
        # 분석할 컬럼 선택 옵션 추가
        st.sidebar.subheader("분석 옵션")
        all_columns = processed_df.columns.tolist()
        selected_columns = st.sidebar.multiselect(
            "분석할 컬럼 선택 (선택하지 않으면 모든 컬럼 사용)",
            options=all_columns,
            default=[]
        )
        
        # 데이터 필터링 옵션 추가
        st.sidebar.subheader("데이터 필터링")
        filter_column = st.sidebar.selectbox(
            "필터링할 컬럼 선택 (선택 사항)",
            options=["없음"] + all_columns
        )
        
        # 선택한 컬럼에 따라 필터 값 선택 UI 표시
        if filter_column != "없음":
            unique_values = processed_df[filter_column].unique()
            if len(unique_values) <= 20:  # 고유 값이 적은 경우 다중 선택 제공
                filter_values = st.sidebar.multiselect(
                    f"{filter_column} 값 선택",
                    options=unique_values,
                    default=[]
                )
                if filter_values:
                    st.session_state.filtered_df = processed_df[processed_df[filter_column].isin(filter_values)]
                else:
                    st.session_state.filtered_df = processed_df
            else:  # 고유 값이 많은 경우 텍스트 검색 제공
                filter_text = st.sidebar.text_input(f"{filter_column} 검색 (부분 일치)")
                if filter_text:
                    st.session_state.filtered_df = processed_df[processed_df[filter_column].astype(str).str.contains(filter_text, case=False)]
                else:
                    st.session_state.filtered_df = processed_df
        else:
            st.session_state.filtered_df = processed_df
        
        # 필터링된 데이터 미리보기
        if filter_column != "없음":
            st.sidebar.subheader("필터링된 데이터")
            st.sidebar.write(f"필터링된 행 수: {len(st.session_state.filtered_df)}")
            st.sidebar.dataframe(st.session_state.filtered_df.head())
        
        # 사용자 노트 입력 필드 추가
        st.sidebar.subheader("분석 노트")
        user_notes = st.sidebar.text_area(
            "데이터에 대한 추가 정보나 분석 지시사항을 입력하세요",
            value=st.session_state.get("user_notes", ""),
            height=100
        )
        st.session_state.user_notes = user_notes
        
        # 설정 저장 및 불러오기 기능
        st.sidebar.subheader("설정 관리")
        
        # 설정 저장
        config_name = st.sidebar.text_input("저장할 설정 이름")
        if st.sidebar.button("현재 설정 저장") and config_name:
            st.session_state.saved_configs[config_name] = {
                "selected_columns": selected_columns if selected_columns else all_columns,
                "filter_column": filter_column,
                "user_notes": user_notes
            }
            
            # 필터 값도 저장
            if filter_column != "없음":
                if len(processed_df[filter_column].unique()) <= 20:
                    st.session_state.saved_configs[config_name]["filter_values"] = filter_values if 'filter_values' in locals() else []
                else:
                    st.session_state.saved_configs[config_name]["filter_text"] = filter_text if 'filter_text' in locals() else ""
            
            st.sidebar.success(f"설정 '{config_name}'이(가) 저장되었습니다.")
        
        # 설정 불러오기
        if st.session_state.saved_configs:
            load_config = st.sidebar.selectbox(
                "저장된 설정 불러오기",
                ["선택하세요..."] + list(st.session_state.saved_configs.keys())
            )
            
            if load_config != "선택하세요..." and st.sidebar.button("설정 불러오기"):
                config = st.session_state.saved_configs[load_config]
                st.session_state.selected_columns = config["selected_columns"]
                st.session_state.user_notes = config["user_notes"]
                st.sidebar.success(f"설정 '{load_config}'이(가) 불러와졌습니다. 페이지를 새로고침하세요.")
        
        # 디버그 모드 옵션 추가
        debug_mode = st.sidebar.checkbox("디버그 모드 (입력 데이터 표시)", value=False)
        st.session_state.debug_mode = debug_mode
        
        # 선택된 컬럼 정보 저장
        if selected_columns:
            st.session_state.selected_columns = selected_columns
        else:
            st.session_state.selected_columns = all_columns
        
        
        # 에이전트에게 파일 내용에 대해 질문하도록 안내
        st.sidebar.info("채팅창에서 업로드한 파일에 대해 질문할 수 있습니다.")

# 채팅 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 채팅 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 파일 데이터가 있는 경우, 프롬프트에 파일 정보와 데이터 추가
    if st.session_state.processed_data is not None:
        # 필터링된 데이터 사용
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            original_df = st.session_state.processed_data
            is_filtered = df.shape[0] != original_df.shape[0]
        else:
            df = st.session_state.processed_data
            is_filtered = False
        
        # 선택된 컬럼만 사용
        if st.session_state.selected_columns:
            selected_cols = st.session_state.selected_columns
            df_selected = df[selected_cols]
        else:
            selected_cols = df.columns.tolist()
            df_selected = df
        
        # 기본 파일 정보 추가
        if is_filtered:
            file_info = f"업로드된 파일 정보: 전체 {original_df.shape[0]}행 중 {df.shape[0]}행 필터링됨, {len(selected_cols)}열, 선택된 컬럼: {', '.join(selected_cols)}"
        else:
            file_info = f"업로드된 파일 정보: {df.shape[0]}행 x {len(selected_cols)}열, 선택된 컬럼: {', '.join(selected_cols)}"
        
        # 실제 데이터 샘플 추가 (처음 10개 행 또는 전체 행이 10개 미만인 경우 전체)
        sample_rows = min(10, df.shape[0])
        data_sample = df_selected.head(sample_rows).to_string(index=False)
        
        # 데이터에 대한 기본 통계 정보 추가
        try:
            numeric_stats = df_selected.describe().to_string()
            stats_info = f"\n\n데이터 통계 정보:\n{numeric_stats}"
        except:
            stats_info = ""
        
        # 데이터 유형 정보 추가
        dtypes_info = "\n\n데이터 유형:\n" + "\n".join([f"{col}: {df_selected[col].dtype}" for col in selected_cols])
        
        # 결측치 정보 추가
        null_counts = df_selected.isnull().sum()
        null_info = "\n\n결측치 정보:\n" + "\n".join([f"{col}: {null_counts[col]}개" for col in selected_cols])
        
        # 사용자 노트 추가
        user_notes_section = ""
        if st.session_state.user_notes.strip():
            user_notes_section = f"\n\n사용자 분석 노트:\n{st.session_state.user_notes}"
        
        # 서비스 정보 추가
        service_info = f"\n\n서비스 정보:\n- 서비스 유형: {service_type}\n- 네트워크 연동 유형: {network_type}\n- 서비스 약어: {service_abbr}\n- NAS 사용 여부: {'예' if nas_usage else '아니오'}"
        
        enhanced_prompt = f"{prompt}\n\n[파일 컨텍스트: {file_info}\n\n데이터 샘플:\n{data_sample}{stats_info}{dtypes_info}{null_info}{user_notes_section}{service_info}]"
    else:
        # 파일이 없는 경우에도 서비스 정보 추가
        service_info = f"\n\n서비스 정보:\n- 서비스 유형: {service_type}\n- 네트워크 연동 유형: {network_type}\n- 서비스 약어: {service_abbr}\n- NAS 사용 여부: {'예' if nas_usage else '아니오'}"
        enhanced_prompt = f"{prompt}{service_info}"
    
    # 디버그 모드가 활성화된 경우 전체 프롬프트 표시
    if st.session_state.debug_mode and st.session_state.processed_data is not None:
        with st.expander("에이전트에 전달되는 입력 데이터", expanded=True):
            st.text(enhanced_prompt)
    
    with st.chat_message("assistant"):
        # Create a placeholder for streaming output
        message_placeholder = st.empty()
        
        # Create a callback handler for streaming
        callback_handler = StreamlitCallbackHandler(message_placeholder)
        
        with st.spinner("답변 생성 중..."):
            # Use streaming callback
            response = create_agent().invoke(
                {"input": enhanced_prompt},
                {"callbacks": [callback_handler]}
            )
            
            # Get the final answer
            answer = response["output"] if isinstance(response, dict) and "output" in response else str(response)
            
            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": answer})
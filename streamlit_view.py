from typing import Any, Dict
import pandas as pd
import re
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from agent import create_agent_graph

def preprocess_dynamic(df: pd.DataFrame) -> list:
    # 1. 컬럼명 표준화
    new_columns = []
    for col in df.columns:
        col_clean = col.strip().lower()                  # 공백 제거 + 소문자
        col_clean = re.sub(r"\s+", "_", col_clean)        # 여러 공백 → "_"
        col_clean = col_clean.replace("%", "pct")         # % → pct
        col_clean = col_clean.replace("(", "").replace(")", "")
        new_columns.append(col_clean)
    df.columns = new_columns

    # 2. 값 전처리 (단위 변환 없음)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()  # 문자열 공백 제거
        df[col] = df[col].fillna("")  # NaN → 빈 문자열

    # 3. 중복 제거
    df = df.drop_duplicates()

    # 4. JSON 변환
    json_data = df.to_dict(orient="records")
    return json_data


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
    st.session_state.service_type = None
if "network_type" not in st.session_state:
    st.session_state.network_type = None
if "service_abbr" not in st.session_state:
    st.session_state.service_abbr = ""
if "nas_usage" not in st.session_state:
    st.session_state.nas_usage = False

# 추가 정보 입력 섹션
st.sidebar.title("서비스 정보 입력")
service_type = st.sidebar.selectbox("서비스 유형", [None, "VM 기반 서비스", "AKS 서비스"], index=0 if st.session_state.service_type is None else 
                                   (1 if st.session_state.service_type == "VM 기반 서비스" else 2))
network_type = st.sidebar.selectbox("네트워크 연동 유형", [None, "공인망 연동", "내부망 연동"], index=0 if st.session_state.network_type is None else 
                                   (1 if st.session_state.network_type == "공인망 연동" else 2))
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
if "json_data" not in st.session_state:
    st.session_state.json_data = None

# 파일 처리 함수
def process_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:  # Excel 파일
            df = pd.read_excel(file)
        
        # preprocess_dynamic 메서드를 사용하여 JSON 형태로 전처리
        json_data = preprocess_dynamic(df)
        
        return df, json_data
    except Exception as e:
        st.sidebar.error(f"파일 처리 중 오류가 발생했습니다: {e}")
        return None, None


# 파일이 업로드되면 처리
if uploaded_file is not None:
    st.sidebar.info(f"파일 '{uploaded_file.name}'이(가) 업로드되었습니다.")
    
    # 파일 처리
    processed_df, json_data = process_file(uploaded_file)
    if processed_df is not None:
        st.session_state.processed_data = processed_df
        st.session_state.json_data = json_data
        
        # 데이터 미리보기
        st.sidebar.subheader("데이터 미리보기")
        st.sidebar.dataframe(processed_df.head())
        
        # 기본 컬럼 정보 설정
        all_columns = processed_df.columns.tolist()
        st.session_state.selected_columns = all_columns
        st.session_state.filtered_df = processed_df
        
        
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
        
        
        # 서비스 정보 추가
        service_info = "\n\n서비스 정보:"
        if service_type is not None:
            service_info += f"\n- 서비스 유형: {service_type}"
        if network_type is not None:
            service_info += f"\n- 네트워크 연동 유형: {network_type}"
        if service_abbr:
            service_info += f"\n- 서비스 약어: {service_abbr}"
        service_info += f"\n- NAS 사용 여부: {'예' if nas_usage else '아니오'}"
        
        # JSON 데이터 추가 (최대 10개 항목)
        json_sample = str(st.session_state.json_data[:10]) if st.session_state.json_data else ""
        
        enhanced_prompt = f"{prompt}\n\n[파일 컨텍스트: {file_info}\n\n데이터 샘플:\n{data_sample}{stats_info}{dtypes_info}{null_info}\n\nJSON 데이터:\n{json_sample}{service_info}]"
    else:
        # 파일이 없는 경우에도 서비스 정보 추가
        service_info = "\n\n서비스 정보:"
        if service_type is not None:
            service_info += f"\n- 서비스 유형: {service_type}"
        if network_type is not None:
            service_info += f"\n- 네트워크 연동 유형: {network_type}"
        if service_abbr:
            service_info += f"\n- 서비스 약어: {service_abbr}"
        service_info += f"\n- NAS 사용 여부: {'예' if nas_usage else '아니오'}"
        enhanced_prompt = f"{prompt}{service_info}"
    
    
    with st.chat_message("assistant"):
        # Create a placeholder for streaming output
        message_placeholder = st.empty()
        
        # Create a callback handler for streaming
        callback_handler = StreamlitCallbackHandler(message_placeholder)
        
        with st.spinner("답변 생성 중..."):
            # Use streaming callback
            response = create_agent_graph().invoke(
                {"input": enhanced_prompt},
                {"callbacks": [callback_handler]}
            )
            
            # Get the final answer - extract only the "answer" field from AgentState
            answer = response.get("answer", "")
            
            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": answer})
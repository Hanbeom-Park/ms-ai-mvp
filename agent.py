import os

from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from template import SYSTEM_MESSAGE, AZURE_RESOURCE_SPEC_TEMPLATE, DETAILED_GUIDELINES_TEMPLATE, GENERAL_GUIDELINES_TEMPLATE

load_dotenv()

AZURE_OPENAI_LLM1 = os.getenv("AZURE_OPENAI_LLM1")
AZURE_OPENAI_LLM2 = os.getenv("AZURE_OPENAI_LLM2")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def detailed_policy_search(query: str) -> str:
    """회사 설계 가이드라인 문서에서 정보를 상세하게 검색합니다.
    이 함수는 실제 스펙을 짜기 위해 관련 정책을 상세하게 검색하는 도구입니다.
    """
    retriever = AzureAISearchRetriever(
        content_key="chunk",
        top_k=3,  # 더 많은 결과를 검색하여 상세한 정보 제공
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
    )
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_LLM2"),
        streaming=False
    )
    prompt = ChatPromptTemplate.from_template(DETAILED_GUIDELINES_TEMPLATE)

    search_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    search_result = search_chain.invoke(query)
    return search_result

@tool
def general_policy_search(query: str) -> str:
    """회사 설계 가이드라인 문서에서 일반적인 정책 정보를 검색합니다.
    이 함수는 단순 문의에 대한 응답으로 물어본 부분에 대해서만 답하는 도구입니다.
    """
    retriever = AzureAISearchRetriever(
        content_key="chunk",
        top_k=1,  # 간단한 문의에 대해 핵심적인 정보만 검색
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
    )
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_LLM2"),
        streaming=False
    )
    prompt = ChatPromptTemplate.from_template(GENERAL_GUIDELINES_TEMPLATE)

    search_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    search_result = search_chain.invoke(query)
    return search_result

@tool
def generate_azure_resource_specs(query: str, search_result: str = None) -> str:
    """검색 결과를 기반으로 Azure 리소스 스펙을 생성합니다.
    이 함수는 회사 가이드라인 검색 결과를 받아 Azure 리소스 스펙 목록을 생성합니다.
    search_result가 제공되지 않으면 자동으로 detailed_policy_search를 호출합니다.
    """
    if search_result is None:
        search_result = detailed_policy_search(query)

    # Generate Azure resource specifications based on guidelines
    # 검색 결과를 LLM에 전달하여 Azure 리소스 스펙 목록을 생성하는 단계
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_LLM1"),
        streaming=False
    )
    resource_spec_prompt = ChatPromptTemplate.from_template(AZURE_RESOURCE_SPEC_TEMPLATE)
    resource_spec_chain = (
            {
                "search_result": lambda x: x,
                "original_query": lambda _: query
            }
            | resource_spec_prompt
            | llm
            | StrOutputParser()
    )
    final_result = resource_spec_chain.invoke(search_result)
    return final_result

def create_agent():
    tools = [detailed_policy_search, general_policy_search, generate_azure_resource_specs]
    llm = AzureChatOpenAI(model=AZURE_OPENAI_LLM1, temperature=0, streaming=True)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,
    )
    return agent_executor
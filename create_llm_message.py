import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

def create_llm_msg(system_prompt: str, sessionHistory: list[BaseMessage]) -> list[BaseMessage]:
    resp = []
    resp.append(SystemMessage(content=system_prompt))
    resp.extend(sessionHistory)
    return resp
import streamlit as st
from langchain_core.messages import SystemMessage, BaseMessage
from create_llm_message import create_llm_msg

class JokeAgent:
    def __init__(self, model):
        self.model = model
        self.system_prompt = """
        You are a Comedian.
        You will be given a topic and you will need to tell a joke about it.
        You will need to tell a joke that is funny and relevant to the topic.
        You will need to tell a joke that is appropriate for all ages.
        You will need to tell a joke that is not too long or too short.
        You will need to tell a joke that is not too offensive.
        You will need to tell a joke that is not too political.
        You will need to tell a joke that is not too religious.
        You will need to tell a joke that is not too sexual.
        You will need to tell a joke that is not too violent.
        You will need to tell a joke that is not too racist.
        You will need to tell a joke that is not making fun of a group or person.
        """
        self.sessionHistory = []

    def get_response(self, user_input: str):
        msg = create_llm_msg(self.system_prompt, self.sessionHistory)
        llm_response = self.model.invoke(msg)

        return llm_response

    def joke_agent(self, user_input: str, session_history=None):
        if session_history is None:
            session_history = []


        from langchain_core.messages import HumanMessage



        messages = []
        messages.append(SystemMessage(content=self.system_prompt))
        messages.extend(session_history)
        messages.append(HumanMessage(content=user_input))

        llm_response = self.model.invoke(messages)

        return {
            "lnode": "joke_agent",
            "responseToUser": llm_response.content,
            "category": "joke",
            "sessionHistory": session_history,
            "user_input": user_input
        }

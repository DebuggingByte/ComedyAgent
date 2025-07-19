import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from joke_agent import JokeAgent
from create_llm_message import create_llm_msg


class State(TypedDict):
    lnode: Optional[str]
    category: Optional[str]
    sessionHistory: List[BaseMessage]
    user_input: str
    responseToUser: Optional[str]


class Category(BaseModel):
    category: str

class Comedian():
    def __init__(self, api_key):
        # Use model from secrets if available, otherwise default to gpt-4o-mini
        model = st.secrets.get("model", "gpt-4o-mini")
        self.model = ChatOpenAI(model=model, api_key=api_key)

        self.joke_agent_class = JokeAgent(self.model)

        workflow = StateGraph(State)

        workflow.add_node("start", self.initial_classifier)
        workflow.add_node("joke", self.joke_agent)
        workflow.add_node("general", self.general_agent)

        workflow.add_edge(START, "start")
        workflow.add_conditional_edges(
            "start",
            self.route_to_agent,
            {
                "joke": "joke",
                "general": "general"
            }
        )
        workflow.add_edge("joke", END)
        workflow.add_edge("general", END)

        self.workflow = workflow.compile()


        
    def route_to_agent(self, state: State) -> str:
        """Route to the appropriate agent based on category."""
        category = state.get("category", "general")
        # Ensure category is one of the valid options
        if category not in ["joke", "general"]:
            category = "general"  # Default to general for unknown categories
        return category

    #Defining the initial_classifier function
    def initial_classifier(self, state: State) -> State:
        """Classify the user input to determine which agent should handle it."""
        
        # First, check if this is a short response that should continue the previous conversation
        user_input = state["user_input"].lower().strip()
        session_history = state.get("sessionHistory", [])
        
        # If it's a short response and we have conversation history, try to determine context
        if len(user_input) <= 10 and session_history:
            # Look at the last assistant message to determine context
            for msg in reversed(session_history):
                if hasattr(msg, 'content') and msg.content:
                    last_response = str(msg.content).lower()
                    # Check if the last response was from a specific subject
                    if "joke" in last_response:
                        category = "joke"
                        break
                    else:
                        category = "general"
                        break
            else:
                category = "general"
        else:
            # Check for intro/greeting keywords first
            intro_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                             "who are you", "what are you", "introduce yourself", "tell me about yourself",
                             "what can you do", "start", "begin", "howdy", "greetings", "yo", "sup"]
            
            if any(keyword in user_input for keyword in intro_keywords):
                category = "general"
            else:
                # Check for joke-related keywords first
                joke_keywords = ["joke", "funny", "humor", "comedy", "comedian", "laugh", "hilarious", "make me laugh"]
                
                # More sophisticated joke detection
                is_joke_request = False
                
                # Check for direct joke keywords
                if any(keyword in user_input for keyword in joke_keywords):
                    is_joke_request = True
                
                # Check for specific patterns
                joke_patterns = [
                    "make a joke", "tell me a joke", "create a joke", "give me a joke",
                    "funny sentence", "surprise me", "make me laugh", "be funny",
                    "joke about", "funny about", "humor about"
                ]
                
                if any(pattern in user_input for pattern in joke_patterns):
                    is_joke_request = True
                
                if is_joke_request:
                    category = "joke"
                else:
                    # Use the AI classifier for longer or standalone inputs
                    classifier_prompt = """
                   You are a comedian.
                   You will be given a topic and you will need to tell a joke about it.
                   You will need to tell a joke that is funny and relevant to the topic.
                   You will need to tell a joke that is appropriate for all ages.
                   You will need to tell a joke that is not too long or too short.
                   You will need to tell a joke that is not too offensive.
                   You will need to tell a joke that is not too political.
                   You will need to tell a joke that is not too sexual.
                    """
                    #Creating the message with user input included - Include session history for context
                    formatted_prompt = classifier_prompt.format(user_input=state["user_input"])
                    msg = create_llm_msg(formatted_prompt, session_history)  # Include session history for context
                    #Invoking the model
                    llm_response = self.model.invoke(msg)
                    #Getting the category
                    category = str(llm_response.content).strip().lower()
                    
                    # Validate and clean the category
                    if category not in ["joke", "general"]:
                        # If the classifier didn't return a valid category, default to general
                        category = "general"
        
        #Returning the state with all original values preserved
        result_state = {
            **state,  # Preserve all original state values
            "category": category,
            "lnode": "initial_classifier"
        }
        return result_state  # type: ignore

    def joke_agent(self, state: State) -> State:
        """Handle joke-related queries."""
        return self.joke_agent_class.joke_agent(state["user_input"], state.get("sessionHistory", []))  # type: ignore
            
    def general_agent(self, state: State) -> State:
        """Handle general queries that don't fit other categories."""
        user_input = state["user_input"].lower()
        
        # Check for introduction/greeting keywords
        intro_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                         "who are you", "what are you", "introduce yourself", "tell me about yourself",
                         "what can you do", "start", "begin", "howdy", "greetings", "yo", "sup"]
        
        is_intro = any(keyword in user_input for keyword in intro_keywords)
        
        if is_intro:
            response = """Hello! I'm your AI Comedian, and I'm here to help you with your jokes! 

Just ask me any question related to jokes, and I'll help you make it a joke in no time. What would you like to work on today?"""
        else:
            response = """I'm sorry, but I can only help with questions related to jokes. 

Your question doesn't seem to fit these subjects. I'm designed to be a focused comedian for jokes only.

Please ask me something about:
• Jokes

What joke would you like to hear?"""
        
        return {
            "lnode": "general_agent",
            "responseToUser": response,
            "category": "general",
            "sessionHistory": state.get("sessionHistory", []),
            "user_input": state["user_input"]
        }   
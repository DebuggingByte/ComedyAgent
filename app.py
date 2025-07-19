import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import Comedian, State

def main():
    st.title("🃏 AI Comedian")
    st.write("Tell me a topic, and I'll help you make it a joke in no time.")

    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    if prompt := st.chat_input("What joke would you like to work on today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})


        with st.chat_message("user"):
            st.markdown(prompt)


        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                agent = Comedian(st.secrets["OPENAI_API_KEY"])


                session_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        session_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        session_history.append(AIMessage(content=msg["content"]))

                initial_state: State = {
                    "user_input": prompt,
                    "sessionHistory": session_history,
                    "lnode": None,
                    "category": None,
                    "responseToUser": None
                }

                final_state = agent.workflow.invoke(initial_state)

                if final_state.get("responseToUser"):
                    response = final_state["responseToUser"]
                    message_placeholder.markdown(response)



                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    message_placeholder.markdown("Sorry, I couldn't generate a joke for that topic.")


            except Exception as e:
                st.error(f"Error: {str(e)}")
                message_placeholder.markdown("Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    main()
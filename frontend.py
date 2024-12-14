import streamlit as st
import requests

st.title("AI Assistant")

url="http://:8000"

with st.sidebar:

    with st.expander("Model Parameters"):

        model = st.selectbox(
            "select model",
            ("Llama3.2-1B","Llama3.2-3B","Qwen2.5-coder-1.5B")
        )

        tools = st.multiselect(
        "Select the tools",
        ["retriever","Wikipedia","DuckDuckGo","PythonREPL","Stock_price"],
        )
        
        col1, col2= st.columns(2)

        with col1:
            reason = st.checkbox("reason")
        with col2:         
            set_model=st.button("set")

        if set_model:        
            payload={"model":model,"tools":tools,"reason":reason}

            st.write(model)
            st.write(tools)
            st.write(reason)

            try:
                response = requests.post(f"{url}/set",json=payload)
                if response.status_code == 200:
                    st.success("All parameter set")
                else:
                    st.error(f"Failed to set: {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            


        uploaded_files = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=True
        )
        content=st.text_area("Details of documents")
        
        index=st.button("index")

        if index:
                    
            files=[("files", (file.name, file, "application/pdf")) for file in uploaded_files]
        
            try:
                response = requests.post(f"{url}/store_docs",files=files)
                st.write(content)
                if response.status_code == 200:
                    st.success("All files uploaded successfully!")                
                else:
                    st.error(f"Failed to upload files: {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"An error occurred hai: {str(e)}")

    st.write("Conversations")

with st.chat_message("assistant"):
    st.markdown("Hello! How can I assist you today?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history=""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
prompt = st.chat_input("Ask your question ?")

previous_conversations=""

if prompt:
    # Display user message in chat message container
    prompt_f=st.session_state.history+" user:"+prompt
    #st.popover(prompt_f)
    payload = {"question": prompt_f}
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add assistant message to chat history
    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        with st.spinner("AI is thinking"):
            try:
                # Open a streaming request to the backend
                response = requests.post(f"{url}/stream", json=payload, stream=True)
                response.raise_for_status()  # Raise error if request fails 15.207.111.61
                
                # Collect and display the streaming output
                ai_response = ""                
                for chunk in response.iter_content(chunk_size=1):
                    if chunk:
                        ai_response += chunk.decode('utf-8')
                        # Update the last assistant message dynamically
                        response_placeholder.markdown(f"{ai_response}")  # Update the placeholder
                # Finalize the assistant message

                st.session_state.history=st.session_state.history+" assistant:"+ai_response
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                st.error(f"Error: {e}") 








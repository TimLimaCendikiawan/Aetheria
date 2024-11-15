from openai import OpenAI
import tiktoken
import streamlit as st
import fitz  # PyMuPDF
import os

# DEFAULT VALUES
DEFAULT_API_KEY = "d5f6fbe6a47ceda89032cbfe7f159b0014a3aec022ec0f5831bc2af7c5df9a80"
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-Vision-Free"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_TOP_P = 1.0
# DEFAULT_TOP_K = 50
# DEFAULT_REP_PENALTY = 1.2

class ConversationManager:
    def __init__(self, api_key=None, base_url=None, model=None, temperature=None, max_tokens=None, token_budget=None):
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.model = model or DEFAULT_MODEL
        self.temperature = temperature or DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.token_budget = token_budget or DEFAULT_TOKEN_BUDGET
        self.top_p = DEFAULT_TOP_P
        # self.top_k = DEFAULT_TOP_K
        # self.rep_penalty = DEFAULT_REP_PENALTY

        self.system_message = "You are a friendly and supportive guide. You answer questions with kindness, encouragement, and patience, always looking to help the user feel comfortable and confident."
        self.conversation_history = [{"role": "system", "content": self.system_message}]

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)

    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) <= 1:
                break
            self.conversation_history.pop(1)

    def chat_completion(self, prompt):
        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                # top_k=self.top_k,
                # repetition_penalty=self.rep_penalty
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        return ai_response

    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]


### Streamlit code ###
st.title("AI Chatbot")

# Initialize the ConversationManager object
if 'chat_manager' not in st.session_state:
    st.session_state['chat_manager'] = ConversationManager()

chat_manager = st.session_state['chat_manager']

# Sidebar widgets for settings
st.sidebar.header("Settings")
chat_manager.model = st.sidebar.text_input("Model Name", chat_manager.model)
chat_manager.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, chat_manager.temperature, 0.01)
chat_manager.max_tokens = st.sidebar.number_input("Max Tokens", value=chat_manager.max_tokens, min_value=1, step=1)
chat_manager.top_p = st.sidebar.slider("Top-p", 0.0, 1.0, chat_manager.top_p, 0.01)
# chat_manager.top_k = st.sidebar.number_input("Top-k", value=chat_manager.top_k, min_value=1, step=1)
# chat_manager.rep_penalty = st.sidebar.slider("Repetition Penalty", 0.0, 2.0, chat_manager.rep_penalty, 0.1)

# File input for PDF
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

# Extract and display content from PDF
if uploaded_file is not None:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        file_content = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            file_content += page.get_text()

    st.write("**File Content:**")
    st.write(file_content)

    # Optionally, use the file content as input for the chatbot
    if st.button("Send File Content to Chatbot"):
        response = chat_manager.chat_completion(file_content)
        st.write("**Chatbot Response:**")
        st.write(response)

# Chat input from the user
user_input = st.chat_input("Write a message")

# Call the chat manager to get a response from the AI
if user_input:
    response = chat_manager.chat_completion(user_input)

# Display the conversation history
for message in chat_manager.conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

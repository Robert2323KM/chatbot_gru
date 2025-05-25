import streamlit as st
from serve_gru import reply

st.set_page_config(page_title="Chatbot GRU", page_icon="🤖")
st.title("💬 Chatbot GRU (Cornell Movie Dialogs)")

# Inicializa historial
if "history" not in st.session_state:
    st.session_state.history = []

# Campo de chat integrado
msg = st.chat_input("Escribe tu mensaje...")
if msg:
    # Añade mensaje del usuario
    st.session_state.history.append(("user", msg))
    # Obtiene respuesta del modelo
    bot_resp = reply(msg)
    st.session_state.history.append(("assistant", bot_resp))

# Renderiza el chat
for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)

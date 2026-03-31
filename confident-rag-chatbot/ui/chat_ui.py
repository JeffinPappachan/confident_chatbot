"""Streamlit chat UI for the Confident Group AI Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import BACKEND_URL


st.set_page_config(page_title="Confident Group AI Assistant", page_icon=":speech_balloon:")
st.title("Confident Group AI Assistant")
st.caption("Ask questions grounded in the ingested Confident Group documents.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I can help answer questions about Confident Group.",
            "sources": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                st.markdown(f"- {source['document']} (page {source['page']})")


def fetch_response(question: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL.rstrip('/')}/chat",
        json={"question": question},
        timeout=180,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise requests.HTTPError(
            f"Backend returned {response.status_code}: {detail}",
            response=response,
        )
    return response.json()


user_question = st.chat_input("Ask a question about Confident Group")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question, "sources": []})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = fetch_response(user_question)
                answer = payload["answer"]
                sources = payload.get("sources", [])
            except requests.RequestException as exc:
                answer = f"I could not complete that request. Details: {exc}"
                sources = []

            st.markdown(answer)
            if sources:
                st.markdown("**Sources**")
                for source in sources:
                    st.markdown(f"- {source['document']} (page {source['page']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

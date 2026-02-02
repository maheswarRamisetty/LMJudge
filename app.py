import streamlit as st
import pandas as pd
from main import evaluate_single

st.set_page_config(
    page_title="JCJS Evaluation Dashboard",
    layout="wide"
)

st.title("📊 JCJS Manual Evaluation System")
st.markdown("Evaluate **one summary at a time**")

# Session state to store history
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("✍️ Input Text")

conversation = st.text_area(
    "Conversation",
    height=200,
    placeholder="Paste full conversation here..."
)

summary = st.text_area(
    "Summary",
    height=150,
    placeholder="Paste generated summary here..."
)

judgment = st.text_area(
    "Judgment / Reference",
    height=150,
    placeholder="Paste human judgment / gold summary here..."
)

if st.button("🚀 Evaluate"):
    if not conversation or not summary or not judgment:
        st.warning("Please fill all fields.")
    else:
        with st.spinner("Computing scores..."):
            scores = evaluate_single(
                summary=summary,
                conversation=conversation,
                judgment=judgment
            )

        st.success("Evaluation completed ")

        st.subheader("🧮 Scores")
        cols = st.columns(7)
        for col, (k, v) in zip(cols, scores.items()):
            col.metric(k, v)

        st.session_state.history.append(scores)

if st.session_state.history:
    st.subheader("📜 Evaluation History")
    df = pd.DataFrame(st.session_state.history)
    df.index += 1
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download History",
        df.to_csv(index_label="Example"),
        file_name="jcjs_manual_results.csv",
        mime="text/csv"
    )
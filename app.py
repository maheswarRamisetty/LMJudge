import streamlit as st
import pandas as pd
import tempfile
from main import evaluate_single
from data_loader import load_csv

st.set_page_config(
    page_title="JCJS Evaluation Dashboard",
    layout="wide"
)

st.title("📊 JCJS Batch Evaluation System")
st.markdown("Upload a **CSV / Excel file** to evaluate summaries")

if "results" not in st.session_state:
    st.session_state.results = []

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        csv_path = file_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path, index=False)
        file_path = csv_path

    summaries, conversations, judgments = load_csv(file_path)

    st.success(f"Loaded {len(summaries)} rows")

    if st.button("🚀 Run Evaluation"):
        with st.spinner("Evaluating summaries..."):
            results = []

            for i in range(len(summaries)):
                scores = evaluate_single(
                    summary=summaries[i],
                    conversation=conversations[i],
                    judgment=judgments[i]
                )
                scores["Row"] = i + 1
                results.append(scores)

            st.session_state.results = results

if st.session_state.results:
    st.subheader("📈 Evaluation Results")

    df_results = pd.DataFrame(st.session_state.results)
    df_results.set_index("Row", inplace=True)

    st.dataframe(df_results, use_container_width=True)

    st.download_button(
        "⬇️ Download Results",
        df_results.to_csv(),
        file_name="jcjs_batch_results.csv",
        mime="text/csv"
    )

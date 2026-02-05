import streamlit as st
import pandas as pd
import tempfile
from main import evaluate_single
from data_loader import load_csv

st.set_page_config(
    page_title="JCJS Evaluation Dashboard",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>Judging LLM as a Judge</h1>",
    unsafe_allow_html=True
)

st.markdown("Upload a **CSV / Excel file** to evaluate Judgement")

if "avg_results" not in st.session_state:
    st.session_state.avg_results = None

def get_metric_insight(metric_name, score):
    insights = {
        "completeness": {
            (0.9, 1.0): "✅ Excellent - Summaries capture nearly all critical information",
            (0.7, 0.9): "👍 Good - Most key points covered, minor details may be missing",
            (0.5, 0.7): "⚠️ Fair - Important information frequently omitted",
            (0.0, 0.5): "❌ Poor - Significant gaps in coverage"
        },
        "accuracy": {
            (0.9, 1.0): "✅ Excellent - Highly accurate with minimal factual errors",
            (0.7, 0.9): "👍 Good - Generally accurate with occasional minor errors",
            (0.5, 0.7): "⚠️ Fair - Noticeable inaccuracies present",
            (0.0, 0.5): "❌ Poor - Frequent factual errors"
        },
        "coherence": {
            (0.9, 1.0): "✅ Excellent - Highly logical and well-structured",
            (0.7, 0.9): "👍 Good - Mostly coherent with minor flow issues",
            (0.5, 0.7): "⚠️ Fair - Some logical inconsistencies",
            (0.0, 0.5): "❌ Poor - Disorganized and hard to follow"
        },
        "relevance": {
            (0.9, 1.0): "✅ Excellent - Highly focused on key information",
            (0.7, 0.9): "👍 Good - Mostly relevant with minor tangents",
            (0.5, 0.7): "⚠️ Fair - Contains some irrelevant content",
            (0.0, 0.5): "❌ Poor - Includes excessive irrelevant information"
        },
        "conciseness": {
            (0.9, 1.0): "✅ Excellent - Optimally concise and efficient",
            (0.7, 0.9): "👍 Good - Generally concise with minor verbosity",
            (0.5, 0.7): "⚠️ Fair - Somewhat wordy or redundant",
            (0.0, 0.5): "❌ Poor - Excessively verbose"
        },
        "fluency": {
            (0.9, 1.0): "✅ Excellent - Natural and well-written",
            (0.7, 0.9): "👍 Good - Mostly fluent with minor awkwardness",
            (0.5, 0.7): "⚠️ Fair - Noticeable grammatical issues",
            (0.0, 0.5): "❌ Poor - Frequent language errors"
        },
        "consistency": {
            (0.9, 1.0): "✅ Excellent - Highly consistent across summaries",
            (0.7, 0.9): "👍 Good - Generally consistent with minor variations",
            (0.5, 0.7): "⚠️ Fair - Noticeable inconsistencies",
            (0.0, 0.5): "❌ Poor - Highly inconsistent"
        }
    }
    
    metric_lower = metric_name.lower()
    for key in insights.keys():
        if key in metric_lower:
            ranges = insights[key]
            for (low, high), message in ranges.items():
                if low <= score <= high:
                    return message
    
    return "ℹ️ No specific insight available for this metric"

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
            all_scores = []

            for i in range(len(summaries)):
                scores = evaluate_single(
                    summary=summaries[i],
                    conversation=conversations[i],
                    judgment=judgments[i]
                )
                all_scores.append(scores)

            df_scores = pd.DataFrame(all_scores)

            avg_scores = df_scores.mean().round(4)
            avg_df = avg_scores.reset_index()
            avg_df.columns = ["Metric", "Average Score"]

            st.session_state.avg_results = avg_df

if st.session_state.avg_results is not None:
    st.subheader("📊 Average Evaluation Scores")

    st.markdown(
        """
        <div style="max-height: 400px; overflow-y: auto;">
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        st.session_state.avg_results,
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Average Scores",
        st.session_state.avg_results.to_csv(index=False),
        file_name="jcjs_average_scores.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("🧠 Insights")
    
    for _, row in st.session_state.avg_results.iterrows():
        metric = row["Metric"]
        score = row["Average Score"]
        insight = get_metric_insight(metric, score)
        
        st.markdown(f"**{metric}** ({score:.4f})")
        st.markdown(f"{insight}")
        st.markdown("")
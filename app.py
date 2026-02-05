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

st.markdown("Upload a **CSV / Excel file** to evaluate summaries")

if "avg_results" not in st.session_state:
    st.session_state.avg_results = None



def get_embedding_similarity_insight(score):
    if 0.75 <= score <= 1.0:
        return "Very Strong - Rare mistakes, covers most facts strongly (Check for bias if >0.8)"
    elif 0.6 <= score < 0.75:
        return "Strong - Reliable fact checking (Accepted)"
    elif 0.45 <= score < 0.6:
        return "Usable - Strict towards facts (Needs constraints)"
    elif 0.3 <= score < 0.45:
        return "Weak Signal - Often wrong (Misleading)"
    else:
        return "Unreliable - Random guessing"


def get_nli_insight(score):
    if 0.75 <= score <= 1.0:
        return "Very Strong logical entailment"
    elif 0.6 <= score < 0.75:
        return "Strong entailment"
    elif 0.45 <= score < 0.6:
        return "Partial entailment"
    elif 0.3 <= score < 0.45:
        return "Weak reasoning"
    else:
        return "Contradictory / random"


def get_cpath_insight(score):
    if 0.75 <= score <= 1.0:
        return "Very Fluent reasoning"
    elif 0.6 <= score < 0.75:
        return "Fluent via logic"
    elif 0.46 <= score < 0.6:
        return "Understandable"
    else:
        return "Confused reasoning"


def get_jscore_insight(score):
    if 0.75 <= score <= 1.0:
        return "Excellent overall judgement"
    elif 0.6 <= score < 0.75:
        return "Good judgement"
    elif 0.46 <= score < 0.6:
        return "Moderate judgement"
    else:
        return "Poor judgement"


def get_accuracy(score):
    if 0.75 <= score <= 1.0:
        return "Covers almost all factual content"
    elif 0.6 <= score < 0.75:
        return "Misses some facts"
    elif 0.3 <= score < 0.6:
        return "Misses important facts"
    else:
        return "Factually incorrect"


def get_completeness(score):
    if 0.70 <= score <= 1.0:
        return "Discourse units well covered"
    elif 0.50 <= score < 0.70:
        return "Entities partially missing"
    elif 0.25 <= score < 0.50:
        return "Poor semantic recall"
    else:
        return "Incomplete summary"


def get_relevance(score):
    if 0.7 <= score <= 1.0:
        return "Highly relevant"
    elif 0.5 <= score < 0.7:
        return "Somewhat relevant"
    elif 0.25 <= score < 0.5:
        return "Weak relevance"
    else:
        return "Mostly lexical match"


def get_clarity(score):
    if 0.75 <= score <= 1.0:
        return "Clear and readable judgement"
    elif 0.50 <= score < 0.75:
        return "Pronoun / density issues"
    else:
        return "Unclear judgement"



def get_metric_insight(metric_name, score):
    metric_lower = metric_name.lower().strip()

    if "embedding" in metric_lower:
        return get_embedding_similarity_insight(score)

    elif "nli" in metric_lower:
        return get_nli_insight(score)

    elif "cpath" in metric_lower or "c-path" in metric_lower:
        return get_cpath_insight(score)

    elif "accuracy" in metric_lower:
        return get_accuracy(score)

    elif "completeness" in metric_lower:
        return get_completeness(score)

    elif "relevance" in metric_lower:
        return get_relevance(score)

    elif "clarity" in metric_lower:
        return get_clarity(score)

    elif "jscore" in metric_lower or "jcjs" in metric_lower:
        return get_jscore_insight(score)

    else:
        return f"Score interpretation not defined"



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
        with st.spinner("Evaluating Judgements..."):
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

    st.dataframe(
        st.session_state.avg_results,
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Average Scores",
        st.session_state.avg_results.to_csv(index=False),
        file_name="jcjs_average_scores.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("Module-wise Insights")

    for _, row in st.session_state.avg_results.iterrows():
        metric = row["Metric"]
        score = row["Average Score"]
        insight = get_metric_insight(metric, score)

        st.markdown(f"### {metric}")
        st.markdown(f"**Average Score:** `{score:.4f}`")
        st.markdown(insight)
        st.markdown("---")

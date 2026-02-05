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
        return "✅ Very Strong - Rare mistakes, covers most facts strongly (Should be enquired for bias if >0.8)"
    elif 0.6 <= score < 0.75:
        return "👍 Strong - Reliable fact checking, facts present mostly (Accepted)"
    elif 0.45 <= score < 0.6:
        return "⚠️ Usable - Gets obvious cases right, strict towards facts (Needs constraints)"
    elif 0.3 <= score < 0.45:
        return "⚡ Weak Signal - Often wrong about presence (Misleads evaluation)"
    else:
        return "❌ Unreliable - Random guessing (Completely unusable)"

def get_nli_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Strong - Rare mistakes, covers most facts strongly (Should be enquired for bias if >0.8)"
    elif 0.6 <= score < 0.75:
        return "👍 Strong - Reliable fact checking, facts present mostly (Accepted)"
    elif 0.45 <= score < 0.6:
        return "⚠️ Usable - Gets obvious cases right, strict towards facts (Needs constraints)"
    elif 0.3 <= score < 0.45:
        return "⚡ Weak Signal - Often wrong about presence (Misleads evaluation)"
    else:
        return "❌ Unreliable - Random guessing (Completely unusable)"

def get_cpath_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Fluent - Over covers mostly"
    elif 0.6 <= score < 0.75:
        return "👍 Fluent - Via logic (Accepted)"
    elif 0.46 <= score < 0.6:
        return "⚠️ Understandable - Moderate quality"
    else:
        return "❌ Confused - Bad quality"

def get_jscore_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Fluent - Over covers mostly"
    elif 0.6 <= score < 0.75:
        return "👍 Fluent - Via logic (Accepted)"
    elif 0.46 <= score < 0.6:
        return "⚠️ Understandable - Moderate quality"
    else:
        return "❌ Confused - Bad quality"

def get_recall_insight(score):
    if 0.7 <= score <= 1.0:
        return "✅ Strong - Rare omissions, covers most facts (Ideal/Acceptable)"
    elif 0.5 <= score < 0.7:
        return "⚠️ Moderate - Captures main facts but misses some details"
    elif 0.3 <= score < 0.5:
        return "⚡ Partial - Finds some facts but inconsistent coverage"
    else:
        return "❌ Incomplete - Misses most facts, under-reporting risk"

def get_precision_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Strong - Rare mistakes, covers most facts strongly"
    elif 0.6 <= score < 0.75:
        return "👍 Strong - Reliable fact checking, facts present mostly (Accepted)"
    elif 0.45 <= score < 0.6:
        return "⚠️ Usable - Gets obvious cases right, strict towards facts"
    elif 0.3 <= score < 0.45:
        return "⚡ Weak Signal - Often wrong about presence"
    else:
        return "❌ Unreliable - Random guessing (Completely unusable)"

def get_f1_insight(score):
    if 0.7 <= score <= 1.0:
        return "✅ Strong - Rare omissions, covers most facts (Ideal)"
    elif 0.5 <= score < 0.7:
        return "⚠️ Moderate - Captures main facts (Acceptable)"
    elif 0.3 <= score < 0.5:
        return "⚡ Partial - Finds some facts (Inconsistent)"
    else:
        return "❌ Incomplete - Misses most facts (Under-reporting risk)"

def get_rouge_insight(score):
    if 0.7 <= score <= 1.0:
        return "✅ Strong - Rare omissions, covers most facts (Ideal)"
    elif 0.5 <= score < 0.7:
        return "⚠️ Moderate - Captures main facts (Acceptable)"
    elif 0.3 <= score < 0.5:
        return "⚡ Partial - Finds some facts (Inconsistent)"
    else:
        return "❌ Incomplete - Misses most facts (Under-reporting risk)"

def get_bleu_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Fluent - Over covers mostly"
    elif 0.6 <= score < 0.75:
        return "👍 Fluent - Via logic (Accepted)"
    elif 0.46 <= score < 0.6:
        return "⚠️ Understandable - Moderate quality"
    else:
        return "❌ Confused - Bad quality"

def get_meteor_insight(score):
    if 0.7 <= score <= 1.0:
        return "✅ Strong - Rare omissions, covers most facts (Ideal)"
    elif 0.5 <= score < 0.7:
        return "⚠️ Moderate - Captures main facts (Acceptable)"
    elif 0.3 <= score < 0.5:
        return "⚡ Partial - Finds some facts (Inconsistent)"
    else:
        return "❌ Incomplete - Misses most facts (Under-reporting risk)"

def get_bertscore_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Strong - Rare mistakes, covers most facts strongly"
    elif 0.6 <= score < 0.75:
        return "👍 Strong - Reliable fact checking, facts present mostly (Accepted)"
    elif 0.45 <= score < 0.6:
        return "⚠️ Usable - Gets obvious cases right, strict towards facts"
    elif 0.3 <= score < 0.45:
        return "⚡ Weak Signal - Often wrong about presence"
    else:
        return "❌ Unreliable - Random guessing (Completely unusable)"

def get_moverscore_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Strong - Rare mistakes, covers most facts strongly"
    elif 0.6 <= score < 0.75:
        return "👍 Strong - Reliable fact checking, facts present mostly (Accepted)"
    elif 0.45 <= score < 0.6:
        return "⚠️ Usable - Gets obvious cases right, strict towards facts"
    elif 0.3 <= score < 0.45:
        return "⚡ Weak Signal - Often wrong about presence"
    else:
        return "❌ Unreliable - Random guessing (Completely unusable)"

def get_bartscore_insight(score):
    if 0.75 <= score <= 1.0:
        return "✅ Very Strong - Rare mistakes, covers most facts strongly"
    elif 0.6 <= score < 0.75:
        return "👍 Strong - Reliable fact checking, facts present mostly (Accepted)"
    elif 0.45 <= score < 0.6:
        return "⚠️ Usable - Gets obvious cases right, strict towards facts"
    elif 0.3 <= score < 0.45:
        return "⚡ Weak Signal - Often wrong about presence"
    else:
        return "❌ Unreliable - Random guessing (Completely unusable)"

def get_accuracy(score):
    if 0.75 <=score <=

def get_metric_insight(metric_name, score):
    metric_lower = metric_name.lower().strip()
    
    if "Embedding" in metric_lower or "similarity" in metric_lower:
        return get_embedding_similarity_insight(score)
    elif "nli" in metric_lower:
        return get_nli_insight(score)
    elif "cpath" in metric_lower or "c-path" in metric_lower:
        return get_cpath_insight(score)
    
    elif "Accuracy" in metric_lower:
        return get_accuracy(score)
    
    elif "jscore" in metric_lower or "j-score" in metric_lower:
        return get_jscore_insight(score)
    elif "recall" in metric_lower:
        return get_recall_insight(score)
    elif "precision" in metric_lower:
        return get_precision_insight(score)
    elif "f1" in metric_lower:
        return get_f1_insight(score)
    elif "rouge" in metric_lower:
        return get_rouge_insight(score)
    elif "bleu" in metric_lower:
        return get_bleu_insight(score)
    elif "meteor" in metric_lower:
        return get_meteor_insight(score)
    elif "bertscore" in metric_lower or "bert" in metric_lower:
        return get_bertscore_insight(score)
    elif "moverscore" in metric_lower or "mover" in metric_lower:
        return get_moverscore_insight(score)
    elif "bartscore" in metric_lower or "bart" in metric_lower:
        return get_bartscore_insight(score)
    
    else:
        return f"ℹ️ Score: {score:.4f} - Metric '{metric_name}' not recognized"

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
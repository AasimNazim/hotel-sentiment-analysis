import pickle
import numpy as np
import scipy.sparse as sp
import streamlit as st
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Setup tokenization packages
nltk.download('punkt', quiet=True)

st.set_page_config(
    page_title="Hotel Sentiment Intelligence",
    layout="wide"  # Layout wide kiya taake metrics aur tables sahi dikhein
)

st.title("Hotel Sentiment Intelligence System")
st.caption(
    "Unified text analysis engine driven by a centralized TF-IDF + XGBoost hybrid inference architecture."
)

# --- SESSION STATE INITIALIZATION FOR BATCH TRACKING ---
if "batch_reviews" not in st.session_state:
    st.session_state.batch_reviews = []  # To store dynamically typed reviews

# --- 1. LOAD NEW MODEL ARTIFACTS ---
@st.cache_resource
def load_classification_pipeline():
    try:
        with open("xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return model, vectorizer, label_encoder
    except FileNotFoundError as e:
        st.error(f"Critical Error: Missing model artifacts. Details: {e}")
        st.stop()

model, vectorizer, label_encoder = load_classification_pipeline()

# --- 2. EXACT LEXICON DICTIONARIES ---
good_keywords = {
    "great", "excellent", "amazing", "awesome", "wonderful", "fantastic", "outstanding", 
    "perfect", "superb", "clean", "spotless", "comfortable", "cozy", "accommodating", "kind", 
    "warm smile", "great breakfast", "great staff", "clean room", "updated", "quiet", "stay again", 
    "go to hotel", "friendly", "helpful", "professional", "courteous", "polite", "attentive", 
    "welcoming", "beautiful", "spacious", "modern", "luxurious", "recommend", "recommended", 
    "enjoyed", "love", "loved", "impressed", "exceptional", "pleasant", "delightful", "convenient", 
    "quick", "efficient", "best", "value", "worth", "affordable", "delicious", "tasty", "fresh", 
    "peaceful", "great service", "excellent service", "friendly staff", "helpful staff", 
    "professional staff", "great location", "perfect location", "comfortable bed", "wonderful stay", 
    "fantastic stay", "pleasant stay", "beautiful view", "highly recommend"
}

average_keywords = {
    "okay", "ok", "fine", "decent", "acceptable", "reasonable", "satisfactory", 
    "normal", "standard", "basic", "nothing special", "not bad", "not great"
}

medium_keywords = {
    "average", "fair", "ordinary", "passable", "mediocre", "lacking", "mixed experience", 
    "could be better", "average stay", "normal stay", "ok stay", "okay stay", "it was fine", 
    "it was okay", "nothing exceptional", "nothing extraordinary", "nothing to complain", 
    "no issues", "no problem", "not too bad", "not too good", "could be improved", "could improve", 
    "needs some improvement", "room for improvement", "somewhat satisfied", "partially satisfied", 
    "mixed feelings", "mixed review", "average experience", "fair experience", "decent enough", 
    "acceptable stay", "a bit expensive", "a bit old", "a bit noisy", "a bit far", 
    "a little disappointed", "slightly disappointed", "not entirely satisfied", "not fully satisfied", 
    "expectations were not met fully"
}

worst_keywords = {
    "suck", "sucks", "smelly", "bugs", "bug", "weired", "stolen", "theft", "robbed", "broken into", 
    "would not", "wouldn't", "could not", "couldn't", "never speak", "manager unavailable", 
    "tired staff", "unhappy", "unsafe", "security issue", "flush", "toilet issue", "parking issue", 
    "towed", "bad", "terrible", "awful", "worst", "horrible", "dirty", "filthy", "unclean", "rude", 
    "unprofessional", "broken", "damaged", "outdated", "old", "worn", "stained", "odor", "smell", 
    "noisy", "noise", "uncomfortable", "slow", "delay", "delayed", "overpriced", "expensive", 
    "problem", "issue", "issues", "disappointed", "disappointing", "waste", "avoid", "poor service", 
    "bad service", "dirty room", "dirty bathroom", "broken elevator", "old furniture", "not recommend", 
    "never again", "waste of money", "terrible experience", "horrible experience", "not good", 
    "not clean", "not comfortable", "not friendly", "not nice", "not amazing", "not perfect", 
    "not helpful", "needs improvement", "some issues", "poor experience", "not satisfied", 
    "very disappointed", "below expectations", "not worth", "bad experience", "would not stay", 
    "wouldn't stay"
}

# --- N-GRAM EXTRACTOR FUNCTION ---
def extract_all_ngrams(words, max_n=5):
    all_phrases = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            all_phrases.append(" ".join(words[i:i+n]))
    return all_phrases

# --- SINGLE STREAMLIT INFERENCE WRAPPER ---
def predict_single_review(text):
    cleaned_text = text.lower()
    tokenized_words = word_tokenize(cleaned_text)
    total_words = len(tokenized_words)
    extracted_live_phrases = extract_all_ngrams(tokenized_words, max_n=5)
    
    c_good = sum(1 for p in extracted_live_phrases if p in good_keywords)
    c_average = sum(1 for p in extracted_live_phrases if p in average_keywords)
    c_medium = sum(1 for p in extracted_live_phrases if p in medium_keywords)
    c_worst = sum(1 for p in extracted_live_phrases if p in worst_keywords)
    
    X_tfidf_live = vectorizer.transform([text])
    live_engineered = sp.csr_matrix([[float(total_words), float(c_good), float(c_average), float(c_medium), float(c_worst)]])
    X_all_features_live = sp.hstack([X_tfidf_live, live_engineered])
    
    predicted_id = model.predict(X_all_features_live)
    sentiment = label_encoder.inverse_transform(predicted_id)[0]
    
    return sentiment, total_words, c_good, c_average, c_medium, c_worst, extracted_live_phrases

# --- CREATE TABS FOR PRESENTATION ---
tab1, tab2 = st.tabs(["✍️ Dynamic Live Tracker", "📊 Bulk CSV Evaluator"])

# ==================== TAB 1: DYNAMIC LIVE TRACKER ====================
with tab1:
    st.subheader("Type & Accumulate Percentage Report")
    st.markdown("Is section mein aap reviews type karte jayenge aur wo temporary database mein save ho kar live percentage add karte rahenge.")
    
    user_review = st.text_area("Review Text Input", placeholder="Type here...", key="single_text_area", height=100)
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        execute_analysis = st.button("Analyze & Save into Report", use_container_width=True)
    with col_btn2:
        clear_report = st.button("Clear Report Data Counters", use_container_width=True)
        
    if clear_report:
        st.session_state.batch_reviews = []
        st.success("Batch memory cleared successfully!")
        st.rerun()

    if execute_analysis and user_review.strip():
        sentiment, total_words, c_good, c_average, c_medium, c_worst, phrases = predict_single_review(user_review)
        
        # Save into session state memory
        st.session_state.batch_reviews.append({"Review": user_review, "Predicted Sentiment": sentiment})
        
        # Render Individual Result
        st.markdown(f"**Current Review Prediction:**")
        if sentiment.upper() == "GOOD": st.success(f"OUTPUT: {sentiment.upper()}")
        elif sentiment.upper() in ["AVERAGE", "MEDIUM", "NEUTRAL"]: st.warning(f"OUTPUT: {sentiment.upper()}")
        else: st.error(f"OUTPUT: {sentiment.upper()}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Tokens", total_words)
        col2.metric("Good Matches", c_good)
        col3.metric("Average Matches", c_average)
        col4.metric("Medium Matches", c_medium)
        col5.metric("Worst Matches", c_worst)

    # Display Accumulated Dynamic Report Summary
    if st.session_state.batch_reviews:
        st.markdown("---")
        st.subheader("📊 Accumulated Batch Testing Report Summary")
        
        df_batch = pd.DataFrame(st.session_state.batch_reviews)
        total_tracked = len(df_batch)
        
        st.metric(label="Total Reviews Processed in Current Session", value=total_tracked)
        
        # Calculate Percentages
        sentiment_counts = df_batch["Predicted Sentiment"].value_counts()
        
        st.markdown("#### Class Wise Percentage Breakdown:")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        classes = ["Good", "Average", "Medium", "Worst"]
        cols_list = [col_m1, col_m2, col_m3, col_m4]
        
        for idx, cls in enumerate(classes):
            count = sentiment_counts.get(cls, 0)
            percentage = (count / total_tracked) * 100 if total_tracked > 0 else 0
            cols_list[idx].metric(label=f"{cls} Ratio", value=f"{percentage:.1f}%", delta=f"{count} Reviews")
            
        st.dataframe(df_batch, use_container_width=True)

# ==================== TAB 2: BULK CSV EVALUATOR (SIR'S FAVORITE) ====================
with tab2:
    st.subheader("Instant 50+ Reviews Bulk Upload")
    st.markdown("Apni 50 reviews wali CSV file yahan drop karein (Column ka naam **review** hona chahiye).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_file = pd.read_csv(uploaded_file)
        
        if "review" not in df_file.columns:
            st.error("Error: CSV file mein 'review' naam ka column hona laazmi hai.")
        else:
            with st.spinner("Processing Bulk Inference via Hybrid Model Pipeline..."):
                final_results = []
                
                for r_text in df_file["review"]:
                    if str(r_text).strip():
                        res = predict_single_review(str(r_text))
                        final_results.append(res[0]) # Append sentiment only
                    else:
                        final_results.append("Empty/Invalid")
                        
                df_file["Model Prediction"] = final_results
                
                st.success("Bulk Evaluation Completed Successfully!")
                
                # Report Metrics
                total_csv = len(df_file)
                csv_counts = df_file["Model Prediction"].value_counts()
                
                st.markdown("### 📈 Bulk Class Percentage Summary")
                c1, c2, c3, c4 = st.columns(4)
                all_cls = ["Good", "Average", "Medium", "Worst"]
                c_cols = [c1, c2, c3, c4]
                
                for idx, cls in enumerate(all_cls):
                    cnt = csv_counts.get(cls, 0)
                    pct = (cnt / total_csv) * 100 if total_csv > 0 else 0
                    c_cols[idx].metric(label=f"Total {cls}", value=f"{pct:.1f}%", delta=f"{cnt} Rows")
                
                st.markdown("#### Full Predicted Data Preview:")
                st.dataframe(df_file, use_container_width=True)
                
                # Download Button for Submission
                csv_data = df_file.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Final Classified Report CSV", data=csv_data, file_name="hotel_predictions_report.csv", mime="text/csv", use_container_width=True)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Add this import statement
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import googleapiclient.discovery
import googleapiclient.errors
import nltk
import math
from tkinter import Tk, Canvas  # Add Tk and Canvas imports

# Disable warning for deprecated use of st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')

# Load RoBERTa sentiment analysis model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCBCl7evYnl_HdlxF9AOtG5FUWGPsn3xyk"  # Replace with your YouTube API key
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

def get_youtube_comments(video_id, max_results=100):
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment_text)
        return comments
    except googleapiclient.errors.HttpError as e:
        st.error(f"Error retrieving comments: {str(e)}")
        return []

def analyze_comments(comments):
    res_youtube = []
    for comment_text in comments:
        try:
            roberta_result = polarity_scores_roberta(comment_text)
            combined_result = {'roberta_neg': roberta_result[0], 'roberta_neu': roberta_result[1], 'roberta_pos': roberta_result[2], 'comment': comment_text}
            res_youtube.append(combined_result)
        except RuntimeError:
            pass

    results_youtube_df = pd.DataFrame(res_youtube)
    return results_youtube_df

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits.detach().numpy()[0]
    scores = softmax(scores)
    return scores

def overall_sentiment_analysis(results_df):
    overall_roberta_scores = results_df[['roberta_neg', 'roberta_neu', 'roberta_pos']].mean()
    
    positive_score = overall_roberta_scores['roberta_pos']
    neutral_score = overall_roberta_scores['roberta_neu']
    negative_score = overall_roberta_scores['roberta_neg']
    
    if positive_score > (neutral_score + negative_score) and neutral_score > negative_score:
        return 90
    elif positive_score > (neutral_score + negative_score) and neutral_score < negative_score:
        return 70
    elif neutral_score > positive_score and neutral_score > negative_score and positive_score > negative_score:
        return 65
    elif positive_score > neutral_score and neutral_score > negative_score:
        return 67
    elif neutral_score > (positive_score + negative_score):
        return 50
    elif neutral_score > (positive_score + negative_score) and negative_score > positive_score:
        return 30
    elif negative_score > (positive_score + neutral_score):
        return 10


def sentiment_label(compound_score):
    if compound_score >= 0.8:
        return "Very Good"
    elif compound_score >= 0.6:
        return "Good"
    elif compound_score >= 0.4:
        return "Moderate"
    elif compound_score >= 0.2:
        return "Bad"
    else:
        return "Very Bad"

# Function to update gauge meter based on sentiment score (0 to 1)
def update_gauge(sentiment_score, canvas):
    angle = 180 * sentiment_score  # Scale sentiment score to angle (0 to 180 degrees)
    canvas.delete("needle")  # Clear previous needle
    needle_start_x = 100  # Adjust X-coordinate for needle base
    needle_start_y = 100  # Adjust Y-coordinate for needle base
    needle_end_x = needle_start_x + 70 * math.cos(math.radians(angle))  # Calculate needle tip X
    needle_end_y = needle_start_y - 70 * math.sin(math.radians(angle))  # Calculate needle tip Y
    canvas.create_line(needle_start_x, needle_start_y, needle_end_x, needle_end_y, width=5, fill="red", tag="needle")

def gauge_meter(value):
    # Define the HTML content with CSS and JS for the gauge meter
    html_code = f"""
    <html>
    <head>
    <style>
    .gauge-container {{
        position: relative;
        width: 300px;
        height: 150px;
        border-radius: 150px 150px 0 0;
        background-color: #f0f0f0;
        overflow: hidden;
    }}
    .gauge-needle {{
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 75px solid #000;
        transform-origin: bottom;
        transform: translateX(-50%) rotate({value * 1.8 - 90}deg);
    }}
    .gauge-needle-pivot {{
        position: absolute;
        top: 100%;
        left: 50%;
        width: 20px;
        height: 20px;
        margin-top: -10px;
        margin-left: -10px;
        border-radius: 50%;
        background-color: #000;
    }}
    .gauge-section {{
        position: absolute;
        width: 50%;
        height: 100%;
        transform-origin: 100% 100%;
    }}
    .gauge-section-very-bad {{
        background-color: #FF0000;
        transform: rotate(0deg);
    }}
    .gauge-section-bad {{
        background-color: #FFA500;
        transform: rotate(36deg);
    }}
    .gauge-section-neutral {{
        background-color: #FFFF00;
        transform: rotate(72deg);
    }}
    .gauge-section-good {{
        background-color: #7FFF00;
        transform: rotate(108deg);
    }}
    .gauge-section-very-good {{
        background-color: #00FF00;
        transform: rotate(144deg);
    }}
    .label {{
        text-align: center;
        font-size: 12px;
        font-weight: bold;
        color: #000;
        position: absolute;
    }}
    </style>
    </head>
    <body>
    <div class="gauge-container">
        <div class="gauge-section gauge-section-very-bad"></div>
        <div class="gauge-section gauge-section-bad"></div>
        <div class="gauge-section gauge-section-neutral"></div>
        <div class="gauge-section gauge-section-good"></div>
        <div class="gauge-section gauge-section-very-good"></div>
        <div class="gauge-needle"></div>
        <div class="gauge-needle-pivot"></div>
    </div>
    </body>
    </html>
    """
    return html_code

def main():
    st.set_page_config(page_title="YouTube Comment Sentiment Analysis", layout="wide")

    st.title("YouTube Comment Sentiment Analysis")

    st.markdown("""
    <style>
        .big-font {{
            font-size: 24px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Settings")
    link = st.sidebar.text_input("Paste the YouTube video link here:")

    if st.sidebar.button("Analyze", key="analyze_button"):
        st.sidebar.info("Please wait while we analyze the comments...")
        if "youtube.com/watch?v=" not in link:
            st.error("Invalid YouTube video link. Please make sure to paste the correct link.")
        else:
            video_id = link.split("v=")[1]
            youtube_comments = get_youtube_comments(video_id)
            if not youtube_comments:
                st.warning("No comments found for this video.")
            else:
                st.success("Comments analyzed successfully!")
                results_youtube_df = analyze_comments(youtube_comments)

                st.markdown("---")
                st.subheader("Overall Sentiment Analysis Visualization")

                # Create overall sentiment pie chart
                labels = ['Negative', 'Neutral', 'Positive']
                colors = ['#FF9999', '#66B2FF', '#99FF99']
                overall_roberta_scores = results_youtube_df[['roberta_neg', 'roberta_neu', 'roberta_pos']].mean()
                fig, ax = plt.subplots()
                ax.pie(overall_roberta_scores, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                st.pyplot(fig)

                st.subheader("Individual Comment Analysis")
                for i, row in results_youtube_df.iterrows():
                    st.write(f"**Comment {i+1}:**")
                    st.write(row['comment'])

                    st.markdown("---")
                    st.write("**Sentiment Analysis Results:**")
                    st.write(f"RoBERTa: Positive={row['roberta_pos']:.2f}, Neutral={row['roberta_neu']:.2f}, Negative={row['roberta_neg']:.2f}")

                    st.markdown("---")

                st.subheader("Overall Sentiment Analysis Gauge Meter")
                overall_sentiment_score = overall_sentiment_analysis(results_youtube_df)  # Replace with actual calculation
                st.write(f"Overall Sentiment Score: {overall_sentiment_score}")
                # Display the gauge meter using HTML with the calculated sentiment score
                st.components.v1.html(gauge_meter(overall_sentiment_score), width=350, height=200)

if __name__ == "__main__":
    main()

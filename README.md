**YouTube Sentiment Analysis**

This project performs sentiment analysis on YouTube comments using natural language processing techniques. The goal is to analyze the sentiment (positive, negative, or neutral) of comments on YouTube videos to gain insights into viewer opinions and feedback.

**Overview**

YouTube is a major platform for sharing and consuming video content, with millions of users expressing their thoughts and opinions through comments. Analyzing the sentiment of these comments can provide valuable insights for content creators, marketers, and researchers.

This project utilizes machine learning models and techniques to automatically classify the sentiment of YouTube comments. It involves data collection, preprocessing, model training, and sentiment prediction.

**Features**

Data Collection: Collect YouTube comments using the YouTube Data API or web scraping techniques.
Sentiment Analysis: Analyze the sentiment of comments using machine learning models such as RoBERTa.
Visualization: Visualize the overall sentiment distribution of comments and display individual comment analysis results.
Interactive Interface: Provide an interactive interface for users to input YouTube video links and view sentiment analysis results.
Dependencies

streamlit: For building interactive web applications.

pandas: For data manipulation and analysis.

matplotlib: For data visualization.

transformers: For using pre-trained language models.

googleapiclient: For interacting with the YouTube Data API.

nltk: For natural language processing tasks.

**Usage**

To run the project:

1.Install the required dependencies by running:

pip install -r requirements.txt

2.Obtain a YouTube API key from the Google Developer Console and replace "your own developer key" in the code with your API key through Google.

3.Run the application using Streamlit:

streamlit run main.py

4.Enter the YouTube video link in the provided input field and click "Analyze" to view sentiment analysis results.



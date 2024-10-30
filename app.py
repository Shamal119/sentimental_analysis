import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.graph_objects as go
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
import plotly.graph_objs as go

# Download NLTK data
try:
   nltk.data.find('tokenizers/punkt')
except LookupError:
   nltk.download('punkt')

def fetch_news(topic, api_key):
   url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}&language=en&sortBy=publishedAt"
   response = requests.get(url)
   return response.json() if response.status_code == 200 else None

def create_wordcloud(texts):
   wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
   return wordcloud

def analyze_sentiment(text):
   sentences = sent_tokenize(text)
   sentiments = [TextBlob(sentence).sentiment for sentence in sentences]
   
   if not sentences:
       return {
           'overall_polarity': 0,
           'overall_subjectivity': 0,
           'sentence_sentiments': []
       }
   
   return {
       'overall_polarity': sum(s.polarity for s in sentiments) / len(sentiments),
       'overall_subjectivity': sum(s.subjectivity for s in sentiments) / len(sentiments),
       'sentence_sentiments': list(zip(sentences, sentiments))
   }

def main():
   st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")
   
   # Custom CSS
   st.markdown("""
       <style>
       .stTextArea textarea {
           font-size: 16px;
       }
       .sentiment-box {
           padding: 20px;
           border-radius: 10px;
           margin: 10px 0;
       }
       .metric-card {
           background-color: #f0f2f6;
           padding: 20px;
           border-radius: 10px;
           text-align: center;
       }
       </style>
   """, unsafe_allow_html=True)

   # Config
   NEWS_API_KEY = st.secrets["news_api_key"]

   # Sidebar
   with st.sidebar:
       st.title(" Analysis Options")
       analysis_mode = st.radio("Choose Mode", ["Custom Text", "News Analysis"])

   if analysis_mode == "Custom Text":
       st.title(" Text Sentiment Analysis")
       
       col1, col2 = st.columns([2, 1])
       
       with col1:
           text_input = st.text_area("Enter your text to analyze:", height=200,
                                   placeholder="Type or paste your text here...")
           
           if st.button("Analyze Text", type="primary", use_container_width=True):
               if text_input:
                   with st.spinner("Analyzing sentiment..."):
                       sentiment = analyze_sentiment(text_input)
                       
                       # Results visualization
                       st.markdown("###  Analysis Results")
                       
                       metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                       
                       with metrics_col1:
                           sentiment_value = sentiment['overall_polarity']
                           sentiment_color = 'red' if sentiment_value < -0.3 else 'green' if sentiment_value > 0.3 else 'orange'
                           st.markdown(f"""
                           <div class="metric-card">
                               <h3>Sentiment Score</h3>
                               <h2 style="color: {sentiment_color}">{sentiment_value:.2f}</h2>
                           </div>
                           """, unsafe_allow_html=True)
                           
                       with metrics_col2:
                           st.markdown(f"""
                           <div class="metric-card">
                               <h3>Subjectivity</h3>
                               <h2>{sentiment['overall_subjectivity']:.2f}</h2>
                           </div>
                           """, unsafe_allow_html=True)
                           
                       with metrics_col3:
                           sentiment_label = "Positive" if sentiment_value > 0.3 else "Negative" if sentiment_value < -0.3 else "Neutral"
                           st.markdown(f"""
                           <div class="metric-card">
                               <h3>Overall Mood</h3>
                               <h2>{sentiment_label}</h2>
                           </div>
                           """, unsafe_allow_html=True)
                       
                       # Sentence-level analysis
                       st.markdown("### 📝 Sentence Analysis")
                       for sentence, sent_sentiment in sentiment['sentence_sentiments']:
                           sent_color = 'red' if sent_sentiment.polarity < -0.3 else 'green' if sent_sentiment.polarity > 0.3 else 'orange'
                           st.markdown(f"""
                           <div style='padding:10px; border-left:5px solid {sent_color}; margin:10px 0;'>
                           {sentence}<br>
                           <small>Sentiment: <span style='color:{sent_color}'>{sent_sentiment.polarity:.2f}</span> | 
                           Subjectivity: {sent_sentiment.subjectivity:.2f}</small>
                           </div>
                           """, unsafe_allow_html=True)
               else:
                   st.warning("Please enter some text to analyze.")

   else:  # News Analysis
       st.title(" News Sentiment Analysis")
       
       topic = st.text_input("Enter topic to analyze:", placeholder="e.g., technology, climate change, sports...")
       col1, col2 = st.columns([1, 2])
       
       with col1:
           news_count = st.slider("Number of articles", 5, 50, 20)
       
       if st.button("Analyze News", type="primary") and topic:
           with st.spinner("Fetching and analyzing news..."):
               news_data = fetch_news(topic, NEWS_API_KEY)
               
               if news_data and news_data.get('articles'):
                   articles = news_data['articles'][:news_count]
                   
                   # Analyze sentiments
                   sentiments = []
                   for article in articles:
                       sentiment = analyze_sentiment(article['title'] + " " + (article['description'] or ""))
                       sentiments.append({
                           'title': article['title'],
                           'sentiment': sentiment['overall_polarity'],
                           'subjectivity': sentiment['overall_subjectivity'],
                           'url': article['url'],
                           'publishedAt': article['publishedAt']
                       })
                   
                   df = pd.DataFrame(sentiments)
                   
                   # Dashboard layout
                   st.markdown("###  Sentiment Overview")
                   metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                   
                   with metrics_col1:
                       st.metric("Average Sentiment", f"{df['sentiment'].mean():.2f}")
                   with metrics_col2:
                       st.metric("Positive Articles", len(df[df['sentiment'] > 0]))
                   with metrics_col3:
                       st.metric("Negative Articles", len(df[df['sentiment'] < 0]))
                   
                   col1, col2 = st.columns(2)
                   
                   with col1:
                       # Sentiment Distribution
                       fig = go.Figure()
                       fig.add_trace(go.Histogram(
                           x=df['sentiment'],
                           nbinsx=20,
                           name='Sentiment Distribution',
                           marker_color='rgb(55, 83, 109)'
                       ))
                       fig.update_layout(
                           title=f"Sentiment Distribution for '{topic}' News",
                           xaxis_title="Sentiment Score",
                           yaxis_title="Number of Articles"
                       )
                       st.plotly_chart(fig, use_container_width=True)
                       
                       # Word cloud
                       st.markdown("### 🔤 Word Cloud")
                       wordcloud = create_wordcloud([a['title'] for a in articles])
                       plt.figure(figsize=(10,5))
                       plt.imshow(wordcloud)
                       plt.axis('off')
                       st.pyplot(plt)
                   
                   with col2:
                       # Positive/Negative News
                       st.markdown("### 📈 Most Positive News")
                       positive_news = df.nlargest(3, 'sentiment')
                       for _, news in positive_news.iterrows():
                           st.markdown(f"""
                           <div style='padding:10px; border-left:5px solid green; margin:10px 0;'>
                           <a href="{news['url']}" target="_blank">{news['title']}</a><br>
                           <small>Sentiment: {news['sentiment']:.2f}</small>
                           </div>
                           """, unsafe_allow_html=True)
                       
                       st.markdown("### 📉 Most Negative News")
                       negative_news = df.nsmallest(3, 'sentiment')
                       for _, news in negative_news.iterrows():
                           st.markdown(f"""
                           <div style='padding:10px; border-left:5px solid red; margin:10px 0;'>
                           <a href="{news['url']}" target="_blank">{news['title']}</a><br>
                           <small>Sentiment: {news['sentiment']:.2f}</small>
                           </div>
                           """, unsafe_allow_html=True)
               else:
                   st.error("Unable to fetch news. Please try again later.")

if __name__ == "__main__":
   main()
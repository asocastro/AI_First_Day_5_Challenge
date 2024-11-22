import os
import openai
import numpy as np
import pandas as pd
import json
import faiss
import warnings
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup
import pickle
import h5py
import time
from fpdf import FPDF

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure you have the necessary NLTK resources
nltk.download('vader_lexicon')

def classify_sentiments_auto(df):
    """
    Automatically detects text columns in a DataFrame and classifies sentiments.
    
    Args:
        df (pd.DataFrame): Input DataFrame with potential text data.
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for sentiment classification.
    """
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Detect text columns
    text_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).mean() > 0.5]
    
    if not text_columns:
        raise ValueError("No text columns detected in the DataFrame.")
    
    # Process each detected text column
    for column in text_columns:
        sentiment_column = f"{column}_Sentiment"
        df[sentiment_column] = df[column].apply(lambda x: classify_sentiment(sia, x))
    
    return df

def classify_sentiment(sia, text):
    """
    Helper function to classify the sentiment of a given text using SIA.
    
    Args:
        sia (SentimentIntensityAnalyzer): SentimentIntensityAnalyzer instance.
        text (str): Input text for sentiment analysis.
        
    Returns:
        str: Sentiment classification - 'Good', 'Bad', or 'Neutral'.
    """
    if not isinstance(text, str):
        return "Neutral"  # Handle non-string values as neutral
    
    sentiment_scores = sia.polarity_scores(text)
    compound = sentiment_scores['compound']
    
    # Classify sentiment based on compound score
    if compound > 0.05:
        return "Good"
    elif compound < -0.05:
        return "Bad"
    else:
        return "Neutral"

def describe_df(df):
    """
    Analyzes the DataFrame structure, generates descriptive statistics and templates,
    and handles scenarios with missing or no columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'paragraph' column containing descriptive templates.
    """
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a valid DataFrame.")
    
    # Get column information
    column_info = df.dtypes
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    # Function to extract unique values for categorical columns
    def get_unique_values(col, max_display=10):
        uniques = df[col].unique().tolist()
        if len(uniques) > max_display:
            return uniques[:max_display] + ["..."]
        return uniques

    # Handle cases where there are no categorical columns
    unique_values = {col: get_unique_values(col) for col in categorical_features} if not categorical_features.empty else {}

    # Handle cases where there are no numerical columns
    numerical_stats = df[numerical_features].describe() if not numerical_features.empty else pd.DataFrame()

    # Template generation function
    def generate_template(df,
                          column_info,
                          categorical_features,
                          numerical_features,
                          unique_values,
                          numerical_stats):
        """
        Generates a descriptive template for the DataFrame based on its structure.
        """
        # Construct a summary of the DataFrame's structure
        column_summary = "Column Names and Data Types:\n"
        for col, dtype in column_info.items():
            column_summary += f" - {col}: {dtype}\n"

        # Unique values for categorical features
        unique_values_str = "Unique Values for Categorical Features:\n"
        if unique_values:
            for col, uniques in unique_values.items():
                unique_values_str += f" - {col}: {uniques}\n"
        else:
            unique_values_str += " - No categorical features found.\n"

        # Descriptive statistics for numerical features
        numerical_stats_str = "Descriptive Statistics for Numerical Features:\n"
        if not numerical_stats.empty:
            for col in numerical_features:
                numerical_stats_str += f" - {col}:\n"
                for stat_name, value in numerical_stats[col].items():
                    numerical_stats_str += f"   - {stat_name}: {value}\n"
        else:
            numerical_stats_str += " - No numerical features found.\n"

        # Define the system prompt
        system_prompt = """You are an intelligent assistant that creates descriptive templates for transforming dataframe rows into coherent paragraphs.
        Analyze the provided dataframe structure and generate a template sentence that includes placeholders for each column.
        Ensure the template is contextually relevant and maintains grammatical correctness."""

        # Define the user prompt
        user_prompt = f"""
        Analyze the following dataframe structure and create a descriptive template with placeholders for each column.

        <column_summary>
        {column_summary}
        </column_summary>

        <unique_values>
        {unique_values_str}
        </unique_values>

        <numerical_stats>
        {numerical_stats_str}
        </numerical_stats>

        Use the exact column names from the column_summary in generating the variable names in the template,
        as they will be populated with the actual values in the dataset.

        Example Template about a Spotify dataset:
        "{{artist}} gained {{streams}} streams in the song '{{song}}' that was a hit in {{date}}."

        Output only the template without any explanation or introduction.
        The template's variables will be dynamically replaced so make sure they're formatted properly.
        """

        # Generate the template
        return f"The dataset includes {{column_summary}} and {{numerical_stats}}."

    # Handle cases where template cannot be generated
    template = generate_template(df,
                                  column_info,
                                  categorical_features,
                                  numerical_features,
                                  unique_values,
                                  numerical_stats)

    # Function to populate the template with row values
    def populate_template(template, row):
        # Convert row to dictionary and replace NaN with 'N/A'
        row_dict = row.to_dict()
        for key, value in row_dict.items():
            if pd.isna(value):
                row_dict[key] = 'N/A'
        try:
            paragraph = template.format(**row_dict)
        except KeyError as e:
            paragraph = f"Error formatting template: Missing key {e}"
        return paragraph

    # Add a 'paragraph' column
    df['paragraph'] = df.apply(lambda row: populate_template(template, row), axis=1)

    return df


def summarize_dataset(df):
    """
    Summarizes the numerical and categorical data in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: A dictionary containing general, numerical, and categorical summaries.
    """
    summary = {}

    # General overview
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a valid DataFrame.")
    
    summary['Shape'] = df.shape
    summary['Data Types'] = df.dtypes.to_dict()

    # Numerical data summary
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numerical_cols.empty:
        numerical_summary = df[numerical_cols].describe().T
        numerical_summary['skewness'] = df[numerical_cols].skew()
        numerical_summary['kurtosis'] = df[numerical_cols].kurt()
        summary['Numerical Summary'] = numerical_summary
    else:
        summary['Numerical Summary'] = "No numerical columns in the DataFrame."

    # Categorical data summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        categorical_summary = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_summary[col] = {
                'Top Categories': value_counts.head(5).to_dict(),
                'Unique Categories': df[col].nunique(),
                'Missing Values': df[col].isnull().sum()
            }
        summary['Categorical Summary'] = categorical_summary
    else:
        summary['Categorical Summary'] = "No categorical columns in the DataFrame."

    return summary


warnings.filterwarnings("ignore")


st.set_page_config(page_title="SentiSense", page_icon="images/logo.jpg", layout="wide")
with st.container():
    c1, c2, c3 = st.columns((1, 3, 1))
    with c2:
        with st.container():
            a, b, c = st.columns((1, 2, 1))
            with b:
                st.image('images/logo.jpg')
   
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
            st.warning('Enter your OpenAI API key to LogIn', icon='⚠️')
        else:
            st.success('Upload your sentiment dataset so I can summarize it!')

            st.title("SentiSense")

            uploaded_file = st.file_uploader("Upload a CSV file to summarize it!", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                df1 = classify_sentiments_auto(df)
                
                #df2 = describe_df(df1)
                
                df3 = summarize_dataset(df1)
                #st.write(df3)

                SYSTEM_PROMPT = '''System Prompt: SentiSense
You are SentiSense, an intelligent assistant designed to analyze and generate human-readable summaries for dataset insights. Your goal is to transform structured dataset metadata into clear, concise, and informative natural language summaries.

Role (R):
You are SentiSense, a sophisticated data analyst assistant specializing in dataset summarization. You focus on providing insights about the structure, content, and key patterns in datasets.

Intent (I):
Your purpose is to:

Analyze the dataset metadata, including shape, data types, numerical summaries, and categorical summaries.
Generate a cohesive summary that captures the dataset’s structure, key statistics, and categorical patterns.
Ensure the output is easily interpretable for non-technical audiences while preserving critical details.
Context (C):
The input contains:

Shape: Dimensions of the dataset.
Data Types: A dictionary showing the column names and their respective data types.
Numerical Summary: A summary of numerical columns, or a note indicating their absence.
Categorical Summary: A detailed breakdown of categorical columns, including top categories, unique category counts, and missing values.
You use this metadata to construct an insightful narrative about the dataset, including relationships, trends, and anomalies where applicable.

Constraints (C):
Focus only on the provided metadata. Do not assume information beyond the input.
If numerical or categorical columns are missing, explicitly state that.
Avoid overly technical language. Summaries should be understandable to both technical and non-technical audiences.
Examples (E):
Example 1:
Input:

json
Copy code
{
  "Shape": [40, 4],
  "Data Types": {
    "Name": "dtype('O')",
    "Sentiment": "dtype('O')",
    "Name_Sentiment": "dtype('O')",
    "Sentiment_Sentiment": "dtype('O')"
  },
  "Numerical Summary": "No numerical columns in the DataFrame.",
  "Categorical Summary": {
    "Name": {
      "Top Categories": {"user_31": 6, "user_25": 5, "user_39": 3, "user_10": 3, "user_3": 2},
      "Unique Categories": 20,
      "Missing Values": "0"
    },
    "Sentiment": {
      "Top Categories": {
        "Terrible experience, keeps breaking down.": 7,
        "Very satisfied with its performance.": 4,
        "Would never buy this brand again.": 3,
        "Complete waste of money, doesn't clean well.": 3,
        "Fantastic product, worth every penny.": 3
      },
      "Unique Categories": 19,
      "Missing Values": "0"
    },
    "Name_Sentiment": {
      "Top Categories": {"Neutral": 40},
      "Unique Categories": 1,
      "Missing Values": "0"
    },
    "Sentiment_Sentiment": {
      "Top Categories": {"Good": 19, "Bad": 17, "Neutral": 4},
      "Unique Categories": 3,
      "Missing Values": "0"
    }
  }
}
Output: The dataset contains 40 rows and 4 columns, all of which are categorical. No numerical columns are present in the dataset.

The categorical columns are:

Name: Represents unique user identifiers, with 20 distinct values. The most frequent category is "user_31," appearing 6 times. There are no missing values in this column.
Sentiment: Captures customer feedback with 19 unique categories. Common sentiments include:
"Terrible experience, keeps breaking down." (7 occurrences)
"Very satisfied with its performance." (4 occurrences)
"Would never buy this brand again." (3 occurrences)
No missing values are present.
Name_Sentiment: A derived column, where all entries are labeled "Neutral." This column has 1 unique category and no missing values.
Sentiment_Sentiment: Represents an overall sentiment analysis with 3 categories:
"Good" (19 occurrences)
"Bad" (17 occurrences)
"Neutral" (4 occurrences)
No missing values are present.
Overall, the dataset appears to be designed for analyzing customer feedback and sentiment patterns. It provides rich categorical data, including user identifiers and sentiment labels, which can be useful for sentiment classification or trend analysis.

Example 2:
Input:

json
Copy code
{
  "Shape": [0, 5],
  "Data Types": {
    "Column1": "dtype('float64')",
    "Column2": "dtype('object')",
    "Column3": "dtype('int64')",
    "Column4": "dtype('object')",
    "Column5": "dtype('category')"
  },
  "Numerical Summary": "No numerical columns in the DataFrame.",
  "Categorical Summary": {}
}
Output: The dataset contains 0 rows and 5 columns, but no data is present.

The column types are as follows:

Column1: Numerical (float64)
Column2: Categorical (object)
Column3: Numerical (int64)
Column4: Categorical (object)
Column5: Categorical (category)
Since the dataset is empty, no further analysis can be performed. Ensure the data source is correctly loaded before proceeding with further analysis.

Example 3:
Input:

json
Copy code
{
  "Shape": [100, 3],
  "Data Types": {
    "Age": "dtype('int64')",
    "Country": "dtype('object')",
    "Feedback": "dtype('object')"
  },
  "Numerical Summary": {
    "Age": {
      "count": 100,
      "mean": 35.6,
      "std": 8.5,
      "min": 18,
      "25%": 30,
      "50%": 35,
      "75%": 40,
      "max": 60,
      "skewness": 0.25,
      "kurtosis": -1.2
    }
  },
  "Categorical Summary": {
    "Country": {
      "Top Categories": {"USA": 45, "UK": 30, "Canada": 15, "Germany": 5, "Australia": 5},
      "Unique Categories": 5,
      "Missing Values": "0"
    },
    "Feedback": {
      "Top Categories": {"Positive": 70, "Neutral": 20, "Negative": 10},
      "Unique Categories": 3,
      "Missing Values": "0"
    }
  }
}
Output: The dataset contains 100 rows and 3 columns, including both numerical and categorical data.

The numerical column:

Age: Ranges from 18 to 60, with a mean of 35.6 years. The data is slightly positively skewed (0.25) and shows mild kurtosis (-1.2).
The categorical columns are:

Country: Includes 5 unique values, with the majority of entries from "USA" (45), followed by "UK" (30). No missing values are present.
Feedback: Captures sentiment with 3 categories:
"Positive" (70 occurrences)
"Neutral" (20 occurrences)
"Negative" (10 occurrences)
No missing values are present.
This dataset is ideal for demographic analysis and feedback sentiment trends.'''
                user_message = f"data prompt: {df3}"
                struct = [{'role': 'system', 'content': SYSTEM_PROMPT}]
                struct.append({"role": "user", "content": user_message})
                chat = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=struct,
                    temperature=0.7
                )
                response = chat.choices[0].message.content
                struct.append({"role": "assistant", "content": response})
                st.write(response)

                
                
            
        
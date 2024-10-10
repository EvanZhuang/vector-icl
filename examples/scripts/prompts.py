from typing import Dict, List
from transformers import AutoTokenizer
import numpy as np
from tqdm import trange, tqdm
import math
import logging
import joblib
import os
import random
from os.path import dirname, basename, join

path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def get_instructions(dataset_name: str):
    if dataset_name in ["stanfordnlp/sst2", "rotten_tomatoes", "stanfordnlp/imdb"]:
        return "the sentiment (positive, negative) of the previous sentence is:"
    elif dataset_name in ["dair-ai/emotion"]:
        return "the sentiment (sadness, joy, love, anger, fear, surprise) of the previous sentence is:"
    elif dataset_name in ["financial_phrasebank"]:
        return "the sentiment (positive, neutral, negative) of the previous sentence is:"
    else:
        raise ValueError("need to set instructions in get_instructions!")


def get_verbalizer(dataset_name):

    VERB_0 = {" negative.": 0, " positive.": 1}
    VERB_1 = {
        " no.": 0,
        " yes.": 1,
    }
    VERB_FFB_0 = {" negative.": 0, " neutral.": 1, " positive.": 2}
    VERB_FFB_1 = {" no.": 0, " maybe.": 1, " yes.": 2}
    # VERB_EMOTION_BINARY = {0: ' Sad.', 1: ' Happy.'}

    # note: verb=1 usually uses yes/no. We don't support this for emotion, since we must specify a value for each of 6 classes
    VERB_EMOTION_0 = {
        " sadness.": 0,
        " joy.": 1,
        " love.": 2,
        " anger.": 3,
        " fear.": 4,
        " surprise.": 5,
    }

    VERB_EMOTION_1 = {
        " sad": 0,
        " happy": 1,
        " love": 2,
        " anger": 3,
        " fear": 4,
        " surprise": 5,
    }

    # VERB_EMOTION_1 = {0: ' No.', 1: ' Maybe.', 2: ' Yes.'}

    VERB_LIST_DEFAULT = [VERB_0, VERB_1]

    DATA_OUTPUT_STRINGS = {
        "stanfordnlp/sst2": VERB_0,
        "rotten_tomatoes": VERB_0,
        "stanfordnlp/imdb": VERB_0,
        "dair-ai/emotion": VERB_EMOTION_0,
        "financial_phrasebank": VERB_FFB_0
    }

    if dataset_name not in DATA_OUTPUT_STRINGS:
        raise ValueError(f"need to set verbalizer in get_verbalizer!")
    return DATA_OUTPUT_STRINGS[dataset_name]
    

def get_prompts(dataset_name: str):
    if dataset_name in ["stanfordnlp/sst2", "rotten_tomatoes", "stanfordnlp/imdb"]:
        return PROMPTS_MOVIE_0
    elif dataset_name in ["dair-ai/emotion"]:
        return PROMPTS_EMOTION_0
    elif dataset_name in ["financial_phrasebank"]:
        return PROMPTS_FINANCE_0
    else:
        raise ValueError("need to set prompt in get_prompts!")


def get_preprompts(dataset_name: str):
    if dataset_name in ["stanfordnlp/sst2", "rotten_tomatoes", "stanfordnlp/imdb"]:
        return PREPROMPTS_MOVIE_0
    elif dataset_name in ["dair-ai/emotion"]:
        return PREPROMPTS_EMOTION_0
    elif dataset_name in ["financial_phrasebank"]:
        return PREPROMPTS_FINANCE_0
    else:
        raise ValueError("need to set prompt in get_prompts!")


PROMPTS_MOVIE_0 = list(
    set(
        [
            # ' What is the sentiment expressed by the reviewer for the movie?',
            # ' Is the movie positive or negative?',
            " The movie is",
            " Positive or Negative? The movie was",
            " The sentiment of the movie was",
            " The plot of the movie was really",
            " The acting in the movie was",
            " I felt the scenery was",
            " The climax of the movie was",
            " Overall I felt the acting was",
            " I thought the visuals were generally",
            " How does the viewer feel about the movie?",
            " What sentiment does the writer express for the movie?",
            " Did the reviewer enjoy the movie?",
            " The cinematography of the film was",
            " I thought the soundtrack of the movie was",
            " I thought the originality of the movie was",
            " I thought the action of the movie was",
            " I thought the pacing of the movie was",
            " I thought the length of the movie was",
            # Chat-GPT-Generated
            # > Generate more prompts for classifying movie review sentiment as Positive or Negative given these examples:
            " The pacing of the movie was",
            " The soundtrack of the movie was",
            " The production design of the movie was",
            " The chemistry between the actors was",
            " The emotional impact of the movie was",
            " The ending of the movie was",
            " The themes explored in the movie were",
            " The costumes in the movie were",
            " The use of color in the movie was",
            " The cinematography of the movie captured",
            " The makeup and hair in the movie were",
            " The lighting in the movie was",
            " The sound design in the movie was",
            " The humor in the movie was",
            " The drama in the movie was",
            " The social commentary in the movie was",
            " The chemistry between the leads was",
            " The relevance of the movie to the current times was",
            " The depth of the story in the movie was",
            " The cinematography in the movie was",
            " The sound design in the movie was",
            " The special effects in the movie were",
            " The characters in the movie were",
            " The plot of the movie was",
            " The script of the movie was",
            " The directing of the movie was",
            " The performances in the movie were",
            " The editing of the movie was",
            " The climax of the movie was",
            " The suspense in the movie was",
            " The emotional impact of the movie was",
            " The message of the movie was",
            " The use of humor in the movie was",
            " The use of drama in the movie was",
            " The soundtrack of the movie was",
            " The visual effects in the movie were",
            " The themes explored in the movie were",
            " The portrayal of relationships in the movie was",
            " The exploration of societal issues in the movie was",
            " The way the movie handles its subject matter was",
            " The way the movie handles its characters was",
            " The way the movie handles its plot twists was",
            " The way the movie handles its narrative structure was",
            " The way the movie handles its tone was",
            " The casting of the film was",
            " The writing of the movie was",
            " The character arcs in the movie were",
            " The dialogue in the movie was",
            " The performances in the movie were",
            " The chemistry between the actors in the movie was",
            " The cinematography in the movie was",
            " The visual effects in the movie were",
            " The soundtrack in the movie was",
            " The editing in the movie was",
            " The direction of the movie was",
            " The use of color in the movie was",
            " The costume design in the movie was",
            " The makeup and hair in the movie were",
            " The special effects in the movie were",
            " The emotional impact of the movie was",
            " The ending of the movie was",
            " The overall message of the movie was",
            " The genre of the movie was well-executed",
            " The casting choices for the movie were well-suited",
            " The humor in the movie was effective",
            " The drama in the movie was compelling",
            " The suspense in the movie was well-maintained",
            " The horror elements in the movie were well-done",
            " The romance in the movie was believable",
            " The action scenes in the movie were intense",
            " The storyline of the movie was engaging",
            # > Generate nuanced prompts for classifying movie review sentiment as Positive or Negative.
            " The movie had some flaws, but overall it was",
            " Although the movie wasn't perfect, I still thought it was",
            " The movie had its ups and downs, but ultimately it was",
            " The movie was a mixed bag, with some parts being",
            " I have mixed feelings about the movie, but on the whole I would say it was",
            " The movie had some redeeming qualities, but I couldn't help feeling",
            " The movie was entertaining, but lacked depth",
            " The movie had a powerful message, but was poorly executed",
            " Despite its flaws, I found the movie to be",
            " The movie was technically impressive, but emotionally unengaging",
            " The movie was thought-provoking, but also frustrating",
            " The movie had moments of brilliance, but was ultimately disappointing",
            " Although the movie had some good performances, it was let down by",
            " The movie had a strong start, but faltered in the second half",
            " The movie was well-made, but ultimately forgettable",
            " The movie was engaging, but also emotionally exhausting",
            " The movie was challenging, but also rewarding",
            " Although it wasn't perfect, the movie was worth watching because of",
            " The movie was a thrilling ride, but also a bit cliché",
            " The movie was visually stunning, but lacked substance",
        ]
    )
)


PREPROMPTS_MOVIE_0 = [
    "You've been tasked with performing sentiment analysis. Sentiment analysis involves determining the sentiment expressed in a piece of text and categorizing it as positive or negative.",
    "Your assignment is to conduct a sentiment analysis task. Sentiment analysis is the process of discerning the sentiment conveyed in text, sorting it into positive or negative categories.",
    "You're requested to carry out a sentiment analysis. Sentiment analysis entails identifying the sentiment expressed in text and classifying it as positive or negative.",
    "Your task is to perform sentiment analysis. Sentiment analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive or negative.",
    "You've been instructed to undertake a sentiment analysis task. Sentiment analysis is the process of evaluating the sentiment expressed in text, categorizing it as positive or negative.",
    "Your responsibility is to perform sentiment analysis. Sentiment analysis entails deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You're required to conduct a sentiment analysis task. Sentiment analysis involves determining the sentiment expressed in a piece of text, categorizing it as positive or negative.",
    "You're asked to perform a sentiment analysis task. Sentiment analysis is the process of evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "You're instructed to carry out sentiment analysis. Sentiment analysis entails analyzing text to determine the sentiment expressed, categorizing it as positive or negative.",
    "Your duty is to undertake a sentiment analysis task. Sentiment analysis is the process of discerning the sentiment conveyed in text and sorting it into positive or negative categories.",
    "You've been assigned to conduct sentiment analysis. Sentiment analysis involves identifying the sentiment expressed in text and classifying it as positive or negative.",
    "Your obligation is to perform a sentiment analysis task. Sentiment analysis entails evaluating the sentiment in text and sorting it into positive or negative categories.",
    "You're required to perform sentiment analysis. Sentiment analysis involves determining the sentiment expressed in text and categorizing it as positive or negative.",
    "You're instructed to undertake sentiment analysis. Sentiment analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it into positive or negative.",
    "You've been tasked with carrying out sentiment analysis. Sentiment analysis entails deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You're asked to conduct a sentiment analysis task. Sentiment analysis is the process of evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "You're requested to perform sentiment analysis. Sentiment analysis involves analyzing text to determine the sentiment expressed, categorizing it as positive or negative.",
    "Your responsibility is to undertake sentiment analysis. Sentiment analysis entails identifying the sentiment expressed in text and classifying it as positive or negative.",
    "You're required to carry out a sentiment analysis task. Sentiment analysis is the process of discerning the sentiment conveyed in text and sorting it into positive or negative categories.",
    "You've been assigned the task of performing sentiment analysis. Sentiment analysis involves determining the sentiment expressed in text and categorizing it as positive or negative.",
    "Your obligation is to conduct sentiment analysis. Sentiment analysis entails evaluating the sentiment in text and sorting it into positive or negative categories.",
    "You're instructed to perform a sentiment analysis task. Sentiment analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it as positive or negative.",
    "You're asked to undertake sentiment analysis. Sentiment analysis involves deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You're requested to carry out sentiment analysis. Sentiment analysis is the process of evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "Your duty is to perform sentiment analysis. Sentiment analysis involves analyzing text to determine the sentiment expressed, categorizing it as positive or negative.",
    "You're required to undertake a sentiment analysis task. Sentiment analysis is the process of identifying the sentiment expressed in text and classifying it as positive or negative.",
    "You're instructed to conduct sentiment analysis. Sentiment analysis entails evaluating the sentiment in text and sorting it into positive or negative categories.",
    "You've been tasked with performing a sentiment analysis task. Sentiment analysis involves determining the sentiment expressed in text and categorizing it as positive or negative.",
    "Your responsibility is to carry out sentiment analysis. Sentiment analysis entails analyzing text to determine the sentiment conveyed, categorizing it as positive or negative.",
    "You're asked to perform sentiment analysis. Sentiment analysis is the process of discerning the sentiment in text and sorting it into positive or negative categories.",
    "You're requested to undertake a sentiment analysis task. Sentiment analysis involves evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "Your task is to conduct sentiment analysis. Sentiment analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it as positive or negative.",
    "You've been instructed to perform sentiment analysis. Sentiment analysis entails deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You're required to carry out sentiment analysis. Sentiment analysis involves identifying the sentiment expressed in text and classifying it as positive or negative.",
    "You're instructed to undertake a sentiment analysis task. Sentiment analysis is the process of evaluating the sentiment in text and sorting it into positive or negative categories.",
    "Your duty is to perform a sentiment analysis task. Sentiment analysis entails analyzing text to determine the sentiment expressed, categorizing it as positive or negative.",
    "You're asked to carry out sentiment analysis. Sentiment analysis involves deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You're requested to perform a sentiment analysis task. Sentiment analysis is the process of evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "Your responsibility is to undertake sentiment analysis. Sentiment analysis entails analyzing text to determine the sentiment conveyed, categorizing it as positive or negative.",
    "You're required to perform sentiment analysis. Sentiment analysis is the process of discerning the sentiment in text and sorting it into positive or negative categories.",
    "You're instructed to conduct a sentiment analysis task. Sentiment analysis involves evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "You've been tasked with undertaking sentiment analysis. Sentiment analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it as positive or negative.",
    "Your obligation is to perform a sentiment analysis task. Sentiment analysis entails deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You're asked to undertake sentiment analysis. Sentiment analysis involves evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "You're requested to conduct a sentiment analysis task. Sentiment analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it as positive or negative.",
    "Your task is to perform sentiment analysis. Sentiment analysis entails deciphering the sentiment in text and sorting it into positive or negative categories.",
    "You've been instructed to carry out a sentiment analysis task. Sentiment analysis involves identifying the sentiment expressed in text and classifying it as positive or negative.",
    "You're required to undertake sentiment analysis. Sentiment analysis is the process of evaluating the sentiment in text and sorting it into positive or negative categories.",
    "You're instructed to perform a sentiment analysis task. Sentiment analysis involves analyzing text to determine the sentiment expressed, categorizing it as positive or negative.",
    "You're asked to carry out a sentiment analysis task. Sentiment analysis is the process of discerning the sentiment in text and sorting it into positive or negative categories.",
    "You're requested to perform sentiment analysis. Sentiment analysis involves evaluating the sentiment expressed in text and classifying it as positive or negative.",
    "Your responsibility is to undertake a sentiment analysis task. Sentiment analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it as positive or negative."
]


PROMPTS_FINANCE_0 = sorted(
    list(
        set(
            [
                " The financial sentiment of this phrase is",
                " The senement of this sentence is",
                " The general tone here is",
                " I feel the sentiment is",
                " The feeling for the economy here was",
                " Based on this the company's outlook will be",
                " Earnings were",
                " Long term forecasts are",
                " Short-term forecasts are",
                " Profits are",
                " Revenue was",
                " Investments are",
                " Financial signals are",
                " All indicators look",
                # Chat-GPT-Generated
                # > Generate more prompts for classifying financial sentences as Positive or Negative given these examples:
                "Overall, the financial outlook seems to be",
                "In terms of financial performance, the company has been",
                "The financial health of the company appears to be",
                "The market reaction to the latest earnings report has been",
                "The company's financial statements indicate that",
                "Investors' sentiment towards the company's stock is",
                "The financial impact of the recent economic events has been",
                "The company's financial strategy seems to be",
                "The financial performance of the industry as a whole has been",
                "The financial situation of the company seems to be",
                # > Generate nuanced prompts for classifying financial sentences as Positive or Negative.
                "Overall, the assessement of the financial performance of the company is",
                "The company's earnings exceeded expectations",
                "The company's revenue figures were",
                "The unexpected financial surprises were",
                "Investments are",
                "Profits were",
                "Financial setbacks were",
                "Investor expectations are",
                "Financial strategy was",
                # > Generate different prompts for classifying financial sentences, that end with "Positive" or "Negative".
                "Based on the latest financial report, the overall financial sentiment is likely to be",
                "The financial health of the company seems to be trending",
                "The company's earnings for the quarter were",
                "Investors' sentiment towards the company's stock appears to be",
                "The company's revenue figures are expected to be",
                "The company's financial performance is expected to have what impact on the market",
                "The latest financial report suggests that the company's financial strategy has been",
            ]
        )
    )
)

PREPROMPTS_FINANCE_0 = [
    "You've been tasked with analyzing financial news. Financial analysis entails deciphering the sentiment expressed in text, categorizing it into positive, neutral, or negative.",
    "Your assignment is to conduct a financial analysis task on financial news. Financial analysis involves discerning the sentiment conveyed in text, sorting it into positive, neutral, or negative categories.",
    "You're requested to analyze financial news. Financial analysis entails identifying the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "Your task is to perform a financial analysis task on financial news. Financial analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You've been instructed to undertake a financial analysis task on financial news. Financial analysis is the process of evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "Your responsibility is to perform financial analysis on financial news. Financial analysis entails deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're required to conduct a financial analysis task on financial news. Financial analysis involves determining the sentiment expressed in text and categorizing it as positive, neutral, or negative.",
    "You're asked to perform a financial analysis task on financial news. Financial analysis is the process of evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're instructed to analyze financial news. Financial analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're requested to conduct a financial analysis task on financial news. Financial analysis is the process of identifying the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You've been assigned to perform a financial analysis task on financial news. Financial analysis involves evaluating the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "Your obligation is to analyze financial news. Financial analysis entails deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're instructed to perform financial analysis on financial news. Financial analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're asked to undertake a financial analysis task on financial news. Financial analysis is the process of deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're requested to perform financial analysis on financial news. Financial analysis involves evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "Your responsibility is to conduct a financial analysis task on financial news. Financial analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're required to analyze financial news. Financial analysis entails identifying the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're instructed to conduct a financial analysis task on financial news. Financial analysis involves evaluating the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're asked to perform a financial analysis task on financial news. Financial analysis is the process of analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're requested to undertake a financial analysis task on financial news. Financial analysis entails deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You've been assigned the task of analyzing financial news. Financial analysis involves evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "Your obligation is to conduct a financial analysis task on financial news. Financial analysis entails analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're instructed to perform a financial analysis task on financial news. Financial analysis is the process of deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're asked to carry out a financial analysis task on financial news. Financial analysis involves evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're requested to perform financial analysis on financial news. Financial analysis entails analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "Your responsibility is to undertake a financial analysis task on financial news. Financial analysis is the process of identifying the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're required to perform a financial analysis task on financial news. Financial analysis involves evaluating the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're instructed to analyze financial news. Financial analysis is the process of deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're asked to undertake a financial analysis task on financial news. Financial analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're requested to perform a financial analysis task on financial news. Financial analysis is the process of evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "Your responsibility is to perform financial analysis on financial news. Financial analysis entails analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're required to conduct a financial analysis task on financial news. Financial analysis is the process of identifying the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're instructed to perform a financial analysis task on financial news. Financial analysis involves evaluating the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're asked to analyze financial news. Financial analysis is the process of deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're requested to undertake a financial analysis task on financial news. Financial analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "Your responsibility is to perform a financial analysis task on financial news. Financial analysis is the process of evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're required to analyze financial news. Financial analysis entails analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're instructed to conduct a financial analysis task on financial news. Financial analysis is the process of identifying the sentiment expressed in text and classifying it as positive, neutral, or negative.",
    "You're asked to perform financial analysis on financial news. Financial analysis involves evaluating the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "You're requested to carry out a financial analysis task on financial news. Financial analysis is the process of deciphering the sentiment in text and sorting it into positive, neutral, or negative categories.",
    "Your responsibility is to undertake a financial analysis task on financial news. Financial analysis involves analyzing text to determine the sentiment conveyed, categorizing it into positive, neutral, or negative.",
    "You're required to perform a financial analysis task on financial news. Financial analysis is the process of evaluating the sentiment expressed in text and classifying it as positive, neutral, or negative."
]

PROMPTS_EMOTION_0 = list(
    set(
        [
            " The emotion of this sentence is:",
            " This tweet contains the emotion",
            " The emotion of this tweet is",
            " I feel this tweet is related to ",
            " The feeling of this tweet was",
            " This tweet made me feel",
            # Chat-GPT-Generated
            # > Generate prompts for classifying tweets based on their emotion (e.g. joy, sadness, fear, etc.). The prompt should end with the emotion.
            " When I read this tweet, the emotion that came to mind was",
            " The sentiment expressed in this tweet is",
            " This tweet conveys a sense of",
            " The emotional tone of this tweet is",
            " This tweet reflects a feeling of",
            " The underlying emotion in this tweet is",
            " This tweet evokes a sense of",
            " The mood conveyed in this tweet is",
            " I perceive this tweet as being",
            " This tweet gives off a feeling of",
            " The vibe of this tweet is",
            " The atmosphere of this tweet suggests a feeling of",
            " The overall emotional content of this tweet is",
            " The affective state expressed in this tweet is",
            # > Generate language model prompts for classifying tweets based on their emotion (e.g. joy, sadness, fear, etc.). The prompt should end with the emotion.
            " Based on the content of this tweet, the emotion I would classify it as",
            " When reading this tweet, the predominant emotion that comes to mind is",
            " This tweet seems to convey a sense of",
            " I detect a feeling of",
            " If I had to categorize the emotion behind this tweet, I would say it is",
            " This tweet gives off a sense of",
            " When considering the tone and language used in this tweet, I would classify the emotion as",
            # > Generate unique prompts for detecting the emotion of a tweet (e.g. joy, sadness, surprise). The prompt should end with the emotion.
            # ' The emotion of this tweet is',
            " The main emotion in this sentence is",
            " The overall tone I sense is",
            " The mood I am in is",
            " Wow this made me feel",
            " This tweet expresses",
        ]
    )
)

PREPROMPTS_EMOTION_0 = [
    "You've been tasked with analyzing online tweets for sentiment analysis. Sentiment analysis involves determining the emotions expressed in a tweet, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "Your assignment is to conduct sentiment analysis on online tweets. Sentiment analysis is the process of deciphering the emotions conveyed in a tweet, sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to analyze online tweets for sentiment analysis. Sentiment analysis entails identifying the emotions expressed in a tweet and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "Your task is to perform sentiment analysis on online tweets. Sentiment analysis involves analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You've been instructed to undertake sentiment analysis of online tweets. Sentiment analysis is the process of evaluating the emotions expressed in a tweet and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "Your responsibility is to perform sentiment analysis on online tweets. Sentiment analysis entails deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're required to conduct sentiment analysis task on online tweets. Sentiment analysis involves determining the emotions expressed in a tweet and categorizing them as sadness, joy, love, anger, fear, or surprise.",
    "You're asked to perform a sentiment analysis task on online tweets. Sentiment analysis is the process of evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to analyze online tweets for sentiment analysis. Sentiment analysis involves analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to conduct a sentiment analysis task on online tweets. Sentiment analysis is the process of identifying the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You've been assigned to perform sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "Your obligation is to analyze online tweets for sentiment analysis. Sentiment analysis entails deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to perform sentiment analysis on online tweets. Sentiment analysis involves analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're asked to undertake a sentiment analysis task on online tweets. Sentiment analysis is the process of deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to perform sentiment analysis on online tweets. Sentiment analysis involves evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "Your responsibility is to conduct a sentiment analysis task on online tweets. Sentiment analysis is the process of analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're required to analyze online tweets for sentiment analysis. Sentiment analysis entails determining the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to conduct a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're asked to perform a sentiment analysis task on online tweets. Sentiment analysis is the process of analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to undertake a sentiment analysis task on online tweets. Sentiment analysis involves deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You've been assigned the task of performing sentiment analysis on online tweets. Sentiment analysis involves evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "Your obligation is to conduct sentiment analysis task on online tweets. Sentiment analysis entails analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to perform a sentiment analysis task on online tweets. Sentiment analysis is the process of deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're asked to carry out a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're requested to perform sentiment analysis on online tweets. Sentiment analysis is the process of analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "Your responsibility is to undertake a sentiment analysis task on online tweets. Sentiment analysis involves deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're required to perform a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to analyze online tweets for sentiment analysis. Sentiment analysis is the process of deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're asked to undertake a sentiment analysis task on online tweets. Sentiment analysis involves analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to perform a sentiment analysis task on online tweets. Sentiment analysis is the process of evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "Your responsibility is to perform sentiment analysis on online tweets. Sentiment analysis involves analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're required to conduct a sentiment analysis task on online tweets. Sentiment analysis is the process of determining the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to perform a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're asked to analyze online tweets for sentiment analysis. Sentiment analysis is the process of analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to undertake a sentiment analysis task on online tweets. Sentiment analysis involves deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "Your responsibility is to perform a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're required to analyze online tweets for sentiment analysis. Sentiment analysis entails analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "You're instructed to conduct a sentiment analysis task on online tweets. Sentiment analysis is the process of determining the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise.",
    "You're asked to perform a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're requested to carry out a sentiment analysis task on online tweets. Sentiment analysis is the process of analyzing tweets to determine the emotions conveyed, categorizing them into sadness, joy, love, anger, fear, or surprise.",
    "Your responsibility is to undertake a sentiment analysis task on online tweets. Sentiment analysis involves deciphering the emotions in tweets and sorting them into sadness, joy, love, anger, fear, or surprise.",
    "You're required to perform a sentiment analysis task on online tweets. Sentiment analysis involves evaluating the emotions expressed in tweets and classifying them as sadness, joy, love, anger, fear, or surprise."
]



def _calc_features_single_prompt(X, y, m, p, past_key_values=None):
    """Calculate features with a single prompt (results get cached)
    preds: np.ndarray[int] of shape (X.shape[0],)
        If multiclass, each int takes value 0, 1, ..., n_classes - 1 based on the verbalizer
    """
    m.prompt = p
    if past_key_values is not None:
        preds = m.predict_with_cache(X, past_key_values)
    else:
        preds = m.predict(X)
    acc = np.mean(preds == y)
    return preds, acc
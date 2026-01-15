# Customer-churn-project


# Intelligent Customer Churn Prediction System  
### Structured ML + Text Sentiment Insight Engine

## 📌 Project Overview
Customer churn is a critical business problem that directly impacts revenue and long-term growth.  
This project builds an **end-to-end intelligent AI/ML system** that predicts customer churn using **structured customer data** and enhances prediction quality by integrating **sentiment insights extracted from unstructured text data**.

The system is designed with **production thinking**, focusing not only on model accuracy but also on:
- Feature engineering
- Error analysis (RCA)
- Model explainability
- Deployment and scalability considerations

---

## 🎯 Problem Statement
To predict whether a customer is likely to churn based on structured customer attributes and improve prediction robustness by incorporating sentiment signals derived from unstructured customer feedback text.

---

## 📊 Datasets Used

### 1️⃣ Structured Dataset (Primary)
**Telco Customer Churn Dataset**  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

**Description:**
- Customer demographics
- Account information
- Service usage
- Billing details  
- Target variable: `Churn` (Yes / No)

---

### 2️⃣ Unstructured Dataset (Text)
**Twitter US Airline Sentiment Dataset**  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment  

**Description:**
- Customer feedback text (`text`)
- Sentiment labels (`positive`, `neutral`, `negative`)

This dataset is used to build a **text sentiment classifier**, whose output is integrated into the churn prediction system.

---

## 🧠 System Architecture (High-Level)


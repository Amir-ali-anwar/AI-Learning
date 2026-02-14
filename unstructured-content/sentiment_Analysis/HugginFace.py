from transformers import pipeline
def Sentiment_analysis(text):
    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()  # 'POSITIVE' or 'NEGATIVE'
    return label

# Get input from user
text = input("Enter text for sentiment analysis: ")

# Analyze and print result
response = Sentiment_analysis(text)
print(f"{text}: The Sentiment is {response}")
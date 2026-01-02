import openai


def Sentiment_analysis(text):
    messages = [
        {
            "role": "system",
            "content": """You are trained to analyze and detect the sentiment of given text.
                          If you're unsure of an answer, you can say "not sure" and recommend users to review manually."""
        },
        {
            "role": "user",
            "content": f"""Analyze the following text and determine if the sentiment is: positive or negative.
                           Return answer in single word as either positive or negative: {text}"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1,
        temperature=0
    )

    response_text = response.choices[0].message.content.strip().lower()
    return response_text

# Get input from user
text = input("Enter text for sentiment analysis: ")

# Analyze and print result
response = Sentiment_analysis(text)
print(f"{text}: The Sentiment is {response}")

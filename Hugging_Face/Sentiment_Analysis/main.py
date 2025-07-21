import transformers
from transformers import pipeline

pipe= pipeline('text-classification')

user_input= input('Enter a sentence for classification')

result= pipe(user_input)

print("Prediction:", result[0]['label'])
print("Confidence Score:", round(result[0]['score'],9))
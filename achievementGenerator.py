import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
from TextClassificationDataset import TCD

#Constants
NUMLABELS = 4


# Use the model to classify a sentence of a user input

#This provide necessary logits for sequence classification, rather than plain BertModel
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUMLABELS)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data = TCD('RawData.JSON', training_set=True) #note: default is true
model.load_state_dict(torch.load('bert_fine_tuned.pt'))
# Make sure that the model is no longer training
model.eval()

print("\nWhat assignment are you working on today? (Press 'q' or 'quit' to exit)")
while(True):
    # Getting user input for the assignment
    sentence = input("You: ")
    if sentence == "quit" or sentence == "q":
        break
    # Tokenize the Sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    # Forward pass through the model, don't need to calculate gradient

    with torch.no_grad():
        outputs = model(**inputs)
    # Get the predicted class probabilities
    predictions = torch.softmax(outputs.logits, dim=1)
    prob = torch.max(predictions)
    # Get the predicted class label (index with the highest probability)
    predicted_class = torch.argmax(predictions, dim=1).item()
    if(prob<0.85):
        print("Word it differently please, try to add keywords like 'essay', 'eigenvalue', 'robotic arm' etc.")
        print("Try Again Please: ")
    else:
        #choose 5 random achievements from the corresponding item
        map = train_data.__getMap__(predicted_class)
        length = len(map)
        indices = np.random.choice(length, 5, replace=False)
        print('\n')
        for i, n in enumerate(indices):
            print(f"{i + 1}. {map[n]}")
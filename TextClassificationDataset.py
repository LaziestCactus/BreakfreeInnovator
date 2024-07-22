import json
import numpy as np
import torch
from torch.utils.data import Dataset

#Create a Text-Classification Dataset that parses our JSON file
class TCD(Dataset):

    #Creates private variable questions and labels, for a random list of train data
    def __init__(self, json_path, training_set=True):
        #pre-processing
        questions = []
        labels = []
        map = {0:[], 1:[], 2:[], 3:[]} #storing the responses/achievements
        with open(json_path, 'r') as file:
            data = json.load(file)
            #go through every subject
            for type in data["assignmentType"]:
                for q in type['questions']:
                    questions.append(q)
                    #using int to represent label for PyTorch to do work
                    labels.append(int(type['tag']))
                for r in type['responses']:
                    map[int(type['tag'])].append(r)
        
        total_questions = len(questions)
        #using numpy to choose random questions and labels to do our training
        train_indices = np.random.choice(total_questions, int(0.8 *total_questions), replace=False)
        self.questions = [questions[i] for i in train_indices]
        self.labels = [labels[i] for i in train_indices]
        self.map = map

    #accessor methods
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]  

    def __getMap__(self, idx):
        return self.map[idx]
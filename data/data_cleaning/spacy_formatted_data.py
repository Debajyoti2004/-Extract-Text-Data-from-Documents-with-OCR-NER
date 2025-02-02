import pandas as pd 
import spacy as sp
from tqdm import tqdm
import random
import pickle


cleanedData = pd.read_csv(r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\csv_files\cleanedData.csv")
group = cleanedData.groupby(by='id')

cards = group.groups.keys()

def spaCyFormattedData(cards):
    allCardsData = []
    for card in tqdm(cards):
        cardData = []
        groupArray = group.get_group(card)[['text','tag']].values

        content=''
        annotations = {'entities':[]}
        start=0
        end=0

        for text,label in tqdm(groupArray):
            text = str(text)
            stringLength = len(text)+1

            start = end
            end = start + stringLength

            if label!='O':
                annot = (start,end-1,label)
                annotations['entities'].append(annot)

            content = content + text + ' '

        cardData = (content,annotations)
        allCardsData.append(cardData)
    
    return allCardsData

allCardsData = spaCyFormattedData(cards)

def save_to_pickle(allCardsData):
    random.shuffle(allCardsData)
    Train_data, Test_data = allCardsData[:240], allCardsData[240:]

    with open(r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\pickle_files\TrainData.pickle", 'wb') as f:
        pickle.dump(Train_data, f)

    with open(r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\pickle_files\TestData.pickle", 'wb') as f:
        pickle.dump(Test_data, f)

    print(f"Train and Test data saved to {r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\pickle_files"} directory")

save_to_pickle(allCardsData=allCardsData)


import numpy as np
import pandas as pd
import cv2
import pytesseract as pt
import spacy
import re
import string
from PIL import Image

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'

    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)

    text = str(txt).lower()
    text = text.translate(tableWhitespace)
    return text.translate(tablePunctuation)

class EntityGroup:
    def __init__(self):
        self.id = 0
        self.text = ''

    def get_group(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id

def parsing_text(text, label):
    if label == 'PHONE':
        text = re.sub(r'\D', '', text.lower())  
    elif label == 'EMAIL':
        allow_special_chars = r'@_.\-'
        text = re.sub(r'[^A-Za-z0-9' + allow_special_chars + r' ]', '', text.lower())  
    elif label == 'WEB':
        allow_special_chars = r':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9' + allow_special_chars + r' ]', '', text.lower())  
    elif label in ('NAME', 'DES'):
        text = re.sub(r'[^a-z]', '', text.lower()) 
        text = text.title()  
    elif label == 'ORG':
        text = re.sub(r'[^a-z0-9 ]', '', text.lower()) 
        text = text.title()  

    return text

def extract_entities(df_info):
    info_array = df_info[['token', 'label']].values
    entities = {
        'NAME': [],
        'ORG': [],
        'DES': [],
        'PHONE': [],
        'EMAIL': [],
        'WEB': []
    }
    
    previous = 'O'  
    for token, label in info_array:
        bio_tag = label[:1] 
        label_tag = label[2:]

        text = parsing_text(token, label_tag)

        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == 'B':
                    entities[label_tag].append(text) 
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag

    return entities



import numpy as np
import pandas as pd
import cv2
import pytesseract as pt
import spacy
import re
import string
from PIL import Image, ImageDraw, ImageFont
from required_fn import parsing_text, cleanText, EntityGroup, extract_entities

import warnings
warnings.filterwarnings('ignore')

group = EntityGroup()

def get_image_bb(image, img_tagging):
    img_bb = image.copy()
    draw = ImageDraw.Draw(img_bb)

    font = ImageFont.truetype("arial.ttf", 24)

    for l, r, t, b, label, token in img_tagging.values:
        draw.rectangle([(l, t), (r, b)], outline='green', width=5)
        draw.text((l, t - 20), label[0], font=font, fill='orange')

    return img_bb

def get_prediction(image, best_NER_path):
    data = pt.image_to_data(image)

    dataList = list(map(lambda x: x.split('\t'), data.split('\n')))
    data_df = pd.DataFrame(dataList[1:], columns=dataList[0])
    data_df.dropna(inplace=True)

    data_df['text'] = data_df['text'].apply(cleanText)

    df_clean = data_df.query('text != ""')
    content = " ".join([w for w in df_clean['text']])
    

    model_ner = spacy.load(best_NER_path)
    doc = model_ner(content)
    doc_json = doc.to_json()
    
    doc_text = doc_json['text']
    dataframe_tag = pd.DataFrame(doc_json['tokens'])
    dataframe_tag['token'] = dataframe_tag[['start', 'end']].apply(lambda x: doc_text[x[0]:x[1]], axis=1)

    right_table = pd.DataFrame(doc_json['ents'])[['start', 'label']]
    dataframe_tag = pd.merge(dataframe_tag, right_table, how='left', on='start')
    
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

    dataframe_tag.fillna('O', inplace=True)
    df_info = pd.merge(df_clean, dataframe_tag[['start', 'token', 'label']], how='inner', on='start')

    bb_df = df_info.query("label != 'O'")
    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(group.get_group)

    bb_df[['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')

    img_tagging = group_tag_img.agg({
        'left': min,
        'right': max,
        'top': min,
        'bottom': max,
        'label': np.unique,
        'token': lambda x: " ".join(x)
    })

    img_bb = get_image_bb(image=image, img_tagging=img_tagging)

    entities = extract_entities(df_info=df_info)

    return img_bb, entities

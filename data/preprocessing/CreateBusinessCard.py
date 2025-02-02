from PIL import Image
import pytesseract as pt
import pandas as pd
import glob
from tqdm import tqdm
import os

imgPaths = glob.glob(r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\Selected\*.jpeg")

def create_business_card(imgPaths):
    allBusinessCard = pd.DataFrame(columns=['id', 'text'])

    for imgPath in tqdm(imgPaths, desc="Processing Business Cards"):
        file_name = os.path.basename(imgPath)

        image = Image.open(imgPath)
        data = pt.image_to_data(image)

        datalist = [row.split('\t') for row in data.split('\n')]
        df = pd.DataFrame(datalist[1:], columns=datalist[0])

        df.dropna(inplace=True)

        df = df[df['conf'].str.isnumeric()]
        df['conf'] = df['conf'].astype(int)

        useFulData = df.query('conf >= 30')

        businessCard = pd.DataFrame({'id': file_name, 'text': useFulData['text']})

        allBusinessCard = pd.concat([allBusinessCard, businessCard], ignore_index=True)

    return allBusinessCard

business_cards_df = create_business_card(imgPaths)

business_cards_df.to_csv("business_cards_extracted.csv", index=False)

print("OCR processing completed! ✅")

from getPrediction import get_prediction
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

best_NER_model_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\Train_NER\output\model-best"
image_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\Selected\030.jpeg"

image = Image.open(image_path)
image_bb,entities = get_prediction(image=image,
                                   best_NER_path=best_NER_model_path)

image_array = np.array(image_bb)
print(entities)
plt.imshow(image_array)
plt.axis('off')  
plt.show()

print(f"All execution suuccessfully completed!")
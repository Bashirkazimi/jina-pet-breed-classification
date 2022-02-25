from jina import Flow, Document
import numpy as np
from PIL import Image


class UI:
    about_block = """

    ### About

    This is a pet breed classification engine using [Jina's neural search framework](https://github.com/jina-ai/jina/).

    - [Repo](https://github.com/bashirkazimi/jina-pet-breed-classification)
    - [Dataset used: Oxford Cats and Dogs Dataset](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset)
    - [Repo for training the classification model](https://github.com/Bashirkazimi/pet-breed-classification)
    - [Repo for the jina executor pushed to Jina Hub](https://github.com/Bashirkazimi/executor-pet-breed-classifier)
    """

    css = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: "#111";
        background-color: "#eee";
    }}
</style>
"""


headers = {"Content-Type": "application/json"}


def convert_file_to_document(query):
    """
    Reads an uploaded image and sets it up as Document object to be processed by Jina.
    """
    image = Image.open(query)
    img_array = np.array(image, dtype=np.uint8)
    doc = Document(tensor=img_array)
    return doc, image 


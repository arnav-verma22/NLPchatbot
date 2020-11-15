import tensorflow as tf
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
import random


with open("intents.json") as file:
    data=json.load(file)




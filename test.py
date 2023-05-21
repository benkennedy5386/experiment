import argparse
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
import datetime
import tensorflow as tf
import sys
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from utils import label_map_util
sys.path.append("D:/models/research/slim")
sys.path.append("D:/models/research/object_detection")
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
import PyPDF2
from fractions import Fraction
import unidecode
import math
import smtplib
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
import smtplib
import yagmail
import math
from pricelist import pricelist
import time
import shutil
import glob
import yagmail
import pathlib


def get_subs(a_dir):

    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

files = get_subs('D:/models/research/object_detection/server/roofing/')
for file in files:

    count = 0
    for path in pathlib.Path('D:/models/research/object_detection/server/roofing/'+file+'/').iterdir():
        if path.is_file():
            count += 1
    print('last is ',count)

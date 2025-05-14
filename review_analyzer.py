# -*- coding: utf-8 -*-
"""
评论主题建模与情感分析
使用BTM（Biterm Topic Model）和BST（Biterm-based Sentiment-Topic Model）
对用户评论进行主题建模和情感分析
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import jieba
import logging
import random
from gensim import corpora, models

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ...（保持现有review_analyzer.py文件内容不变）...
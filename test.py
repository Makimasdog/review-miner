# -*- coding: utf-8 -*-
"""
评论主题建模与情感分析示例
使用BTM（Biterm Topic Model）和BST（Biterm-based Sentiment-Topic Model）
对用户评论进行主题建模和情感分析
"""

import os
import logging
from review_analyzer import ReviewParser, BTM, BST

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ...（保持现有test.py文件内容不变）...
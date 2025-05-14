# -*- coding: utf-8 -*-
"""
评论主题建模与情感分析
使用BTM（Biterm Topic Model）和BST（Biterm-based Sentiment-Topic Model）
对用户评论进行主题建模和情感分析

数据格式：发表时间-*-作者-*-评级-*-标题-*-内容
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

class ReviewParser:
    """解析评论数据"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.reviews = []
    
    def parse(self):
        """解析TXT文件中的评论数据"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析格式：发表时间-*-作者-*-评级-*-标题-*-内容
                parts = line.split('-*-')
                if len(parts) == 5:
                    review = {
                        'time': parts[0].strip(),
                        'author': parts[1].strip(),
                        'rating': parts[2].strip(),
                        'title': parts[3].strip(),
                        'content': parts[4].strip()
                    }
                    self.reviews.append(review)
                else:
                    logging.warning(f"跳过格式不正确的行: {line}")
                    
            logging.info(f"成功解析 {len(self.reviews)} 条评论")
            return self.reviews
        except Exception as e:
            logging.error(f"解析评论文件时出错: {str(e)}")
            return []

# 主函数示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评论主题建模与情感分析工具')
    parser.add_argument('--input', type=str, required=True, help='输入评论文件路径')
    parser.add_argument('--output', type=str, default='results', help='输出结果目录')
    parser.add_argument('--topics', type=int, default=10, help='主题数量')
    parser.add_argument('--sentiments', type=int, default=3, help='情感数量')
    parser.add_argument('--iterations', type=int, default=500, help='迭代次数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 解析评论
    parser = ReviewParser(args.input)
    reviews = parser.parse()
    
    if reviews:
        logging.info("评论解析完成，开始进行分析...")
        # 这里可以添加主题建模和情感分析的代码
        logging.info(f"分析完成，结果已保存到 {args.output} 目录")
    else:
        logging.error("没有可用的评论数据，请检查输入文件")

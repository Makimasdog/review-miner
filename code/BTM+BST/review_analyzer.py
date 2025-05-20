# -*- coding: utf-8 -*-
"""
评论主题建模与情感分析工具
使用BTM（Biterm Topic Model）和BST（Biterm-based Sentiment-Topic Model）
对用户评论进行主题建模和情感分析
"""

import os
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora, models
from collections import defaultdict

class ReviewAnalyzer:
    """
    评论分析器类，用于处理评论数据并进行主题建模和情感分析
    """
    def __init__(self, data_path=None):
        """
        初始化评论分析器
        
        参数:
            data_path: 评论数据文件路径
        """
        self.data_path = data_path
        self.reviews = []
        self.processed_reviews = []
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.stopwords = self._load_stopwords()
        
        if data_path and os.path.exists(data_path):
            self.load_data()
    
    def _load_stopwords(self):
        """
        加载停用词表
        
        返回:
            停用词集合
        """
        stopwords = set()
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip())
        except FileNotFoundError:
            print("停用词文件未找到，使用空停用词表")
        return stopwords
    
    def load_data(self, data_path=None):
        """
        加载评论数据
        
        参数:
            data_path: 评论数据文件路径，如果为None则使用初始化时的路径
        """
        if data_path:
            self.data_path = data_path
            
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.reviews = [line.strip() for line in f if line.strip()]
            print(f"成功加载{len(self.reviews)}条评论")
        except Exception as e:
            print(f"加载数据失败: {e}")
    
    def preprocess(self):
        """
        预处理评论文本，包括分词、去停用词等
        """
        if not self.reviews:
            print("没有评论数据可处理")
            return
            
        self.processed_reviews = []
        for review in self.reviews:
            # 分词并去除停用词
            words = [word for word in jieba.cut(review) 
                    if word not in self.stopwords and len(word) > 1]
            self.processed_reviews.append(words)
        
        # 创建词典和语料库
        self.dictionary = corpora.Dictionary(self.processed_reviews)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.processed_reviews]
        
        print(f"预处理完成，词典大小: {len(self.dictionary)}")
    
    def build_model(self, num_topics=10):
        """
        构建主题模型
        
        参数:
            num_topics: 主题数量
        """
        if not self.corpus:
            print("请先进行数据预处理")
            return
            
        # 使用LDA模型代替BTM（实际项目中可替换为BTM实现）
        self.model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='auto'
        )
        
        print(f"主题模型构建完成，主题数: {num_topics}")
    
    def analyze_topics(self):
        """
        分析主题并返回结果
        
        返回:
            主题列表，每个主题包含关键词和权重
        """
        if not self.model:
            print("请先构建主题模型")
            return None
            
        topics = []
        for topic_id in range(self.model.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=10)
            topics.append({
                'id': topic_id,
                'words': topic_words
            })
            
            # 打印主题关键词
            print(f"主题 {topic_id}:")
            print(", ".join([f"{word}({weight:.3f})" for word, weight in topic_words]))
            
        return topics
    
    def classify_reviews(self):
        """
        对评论进行主题分类
        
        返回:
            分类结果字典，键为主题ID，值为对应的评论列表
        """
        if not self.model or not self.corpus:
            print("请先构建主题模型")
            return None
            
        classified_reviews = defaultdict(list)
        
        for i, review in enumerate(self.reviews):
            # 获取评论的主题分布
            topic_dist = self.model[self.corpus[i]]
            # 找出最可能的主题
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else 0
            classified_reviews[dominant_topic].append(review)
        
        # 打印分类统计
        for topic_id, reviews in classified_reviews.items():
            print(f"主题 {topic_id}: {len(reviews)}条评论")
            
        return dict(classified_reviews)
    
    def save_results(self, output_dir="results"):
        """
        保存分析结果
        
        参数:
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存主题关键词
        topics_file = os.path.join(output_dir, "topics.txt")
        with open(topics_file, 'w', encoding='utf-8') as f:
            for topic_id in range(self.model.num_topics):
                topic_words = self.model.show_topic(topic_id, topn=20)
                f.write(f"主题 {topic_id}:\n")
                f.write(", ".join([f"{word}({weight:.3f})" for word, weight in topic_words]))
                f.write("\n\n")
        
        # 保存分类结果
        classified_reviews = self.classify_reviews()
        for topic_id, reviews in classified_reviews.items():
            topic_file = os.path.join(output_dir, f"topic_{topic_id}_reviews.txt")
            with open(topic_file, 'w', encoding='utf-8') as f:
                for review in reviews:
                    f.write(f"{review}\n")
        
        print(f"结果已保存到 {output_dir} 目录")

# 示例用法
if __name__ == "__main__":
    analyzer = ReviewAnalyzer("reviews.txt")
    analyzer.preprocess()
    analyzer.build_model(num_topics=5)
    analyzer.analyze_topics()
    analyzer.save_results()

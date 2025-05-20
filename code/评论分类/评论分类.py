# -*- coding: utf-8 -*-
"""
评论分类模块
用于对用户评论进行分类，识别不同类别的评论内容
"""

import os
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class ReviewClassifier:
    """
    评论分类器类，用于训练和预测评论类别
    """
    def __init__(self):
        """
        初始化评论分类器
        """
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()
        self.stopwords = self._load_stopwords()
        self.categories = []
        
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
    
    def preprocess_text(self, text):
        """
        预处理文本
        
        参数:
            text: 待处理的文本
            
        返回:
            处理后的文本
        """
        # 分词
        words = jieba.cut(text)
        # 去除停用词
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 1]
        # 重新组合成文本
        return ' '.join(filtered_words)
    
    def load_data(self, data_path, category_col='category', text_col='review'):
        """
        加载数据
        
        参数:
            data_path: 数据文件路径（CSV格式）
            category_col: 类别列名
            text_col: 文本列名
            
        返回:
            处理后的数据DataFrame
        """
        try:
            data = pd.read_csv(data_path)
            print(f"成功加载数据，共{len(data)}条记录")
            
            # 获取所有类别
            self.categories = data[category_col].unique().tolist()
            print(f"类别列表: {self.categories}")
            
            # 预处理文本
            data['processed_text'] = data[text_col].apply(self.preprocess_text)
            
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def train(self, data, category_col='category', text_col='processed_text', test_size=0.2):
        """
        训练分类器
        
        参数:
            data: 训练数据DataFrame
            category_col: 类别列名
            text_col: 处理后的文本列名
            test_size: 测试集比例
            
        返回:
            训练结果报告
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            data[text_col], data[category_col], test_size=test_size, random_state=42
        )
        
        # 特征提取
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # 训练分类器
        self.classifier.fit(X_train_tfidf, y_train)
        
        # 预测
        y_pred = self.classifier.predict(X_test_tfidf)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"分类准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 可视化结果
        self._plot_results(report)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'test_data': (X_test, y_test),
            'predictions': y_pred
        }
    
    def predict(self, texts):
        """
        预测文本类别
        
        参数:
            texts: 文本列表
            
        返回:
            预测的类别列表
        """
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 特征提取
        X_tfidf = self.vectorizer.transform(processed_texts)
        
        # 预测
        predictions = self.classifier.predict(X_tfidf)
        
        return predictions
    
    def _plot_results(self, report):
        """
        可视化分类结果
        
        参数:
            report: 分类报告字典
        """
        # 提取每个类别的精确率、召回率和F1值
        categories = []
        precision = []
        recall = []
        f1_score = []
        
        for category, metrics in report.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                categories.append(category)
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1_score.append(metrics['f1-score'])
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Category': categories,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        
        # 精确率
        plt.subplot(1, 3, 1)
        sns.barplot(x='Category', y='Precision', data=df)
        plt.title('Precision by Category')
        plt.xticks(rotation=45)
        
        # 召回率
        plt.subplot(1, 3, 2)
        sns.barplot(x='Category', y='Recall', data=df)
        plt.title('Recall by Category')
        plt.xticks(rotation=45)
        
        # F1值
        plt.subplot(1, 3, 3)
        sns.barplot(x='Category', y='F1-Score', data=df)
        plt.title('F1-Score by Category')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('classification_results.png')
        plt.close()
    
    def save_model(self, model_path='trained_models'):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径
        """
        import pickle
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # 保存向量化器
        with open(os.path.join(model_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # 保存分类器
        with open(os.path.join(model_path, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # 保存类别列表
        with open(os.path.join(model_path, 'categories.txt'), 'w', encoding='utf-8') as f:
            for category in self.categories:
                f.write(f"{category}\n")
        
        print(f"模型已保存到 {model_path} 目录")
    
    def load_model(self, model_path='trained_models'):
        """
        加载模型
        
        参数:
            model_path: 模型加载路径
        """
        import pickle
        
        try:
            # 加载向量化器
            with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # 加载分类器
            with open(os.path.join(model_path, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            
            # 加载类别列表
            self.categories = []
            with open(os.path.join(model_path, 'categories.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    self.categories.append(line.strip())
            
            print(f"模型已加载，支持的类别: {self.categories}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

# 示例用法
if __name__ == "__main__":
    classifier = ReviewClassifier()
    # 假设有一个CSV文件包含评论和类别
    data = classifier.load_data('reviews_with_categories.csv')
    if data is not None:
        results = classifier.train(data)
        classifier.save_model()
        
        # 测试预测
        test_reviews = [
            "这个应用非常好用，界面设计很漂亮",
            "经常崩溃，无法正常使用",
            "希望能增加更多功能"
        ]
        predictions = classifier.predict(test_reviews)
        for review, prediction in zip(test_reviews, predictions):
            print(f"评论: {review}")
            print(f"预测类别: {prediction}\n")

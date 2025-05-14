# 评论主题建模与情感分析工具

## 项目简介

这是一个用于分析用户评论的工具，使用BTM（Biterm Topic Model）和BST（Biterm-based Sentiment-Topic Model）对用户评论进行主题建模和情感分析。该工具可以帮助企业或研究人员从大量评论中提取关键主题和情感倾向，从而更好地理解用户反馈。

## 环境要求

- Python 3.8
- 依赖库：numpy, pandas, matplotlib, scikit-learn, jieba, gensim

## 安装方法

1. 克隆仓库到本地
```bash
git clone https://github.com/Makimasdog/review-miner.git
cd review-miner
```

2. 安装依赖包
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备评论数据文件（TXT格式）
2. 运行分析脚本
```bash
python review_analyzer.py --input reviews.txt --output results/
```
3. 查看分析结果

也可以使用训练好的模型直接进行分析：
```bash
python trained_models/use_models.py --input reviews.txt
```

## 输入数据格式

评论数据应为TXT文件，每行一条评论，格式为：

```
发表时间-*-作者-*-评级-*-标题-*-内容
```

例如：

```
2023-01-01-*-用户1-*-5-*-非常满意-*-这个产品非常好用，界面设计简洁明了，功能齐全，操作简单，很满意这次购买。
```

## 功能特点

- 主题建模：使用BTM算法提取评论中的主要话题
- 情感分析：结合主题和情感进行分析，了解用户对不同主题的情感倾向
- 可视化结果：生成直观的图表展示分析结果
- 批量处理：支持批量处理大量评论数据

## 项目结构

```
├── README.md           # 项目说明文档
├── requirements.txt    # 依赖包列表
├── review_analyzer.py  # 主分析程序
├── reviews.txt         # 示例评论数据
├── test.py            # 测试脚本
├── results/           # 分析结果输出目录
└── trained_models/    # 预训练模型目录
    ├── README.md      # 模型使用说明
    ├── run_analysis.bat # Windows批处理脚本
    └── use_models.py  # 使用预训练模型的脚本
```

## 许可证

MIT

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
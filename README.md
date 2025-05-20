# 评论主题建模与情感分析工具
## 安装方法

1. 克隆仓库到本地
```bash
git clone https://github.com/Makimasdog/review-miner.git
cd review-miner
```

# 评论主题建模与情感分析工具
## 项目结构
```
├── Dataset/               # 原始数据及分析结果
├── code/                  # 核心算法实现
│   ├── BTM+BST/          # 主题模型相关代码
│   ├── 聚类/            # 聚类相关代码
│   └── 评论分类/        # 评论分类相关代码
├── test/                  # 测试用例及示例输出
└── requirements.txt       # Python依赖库
```

2. 安装依赖包
```bash
pip install -r requirements.txt
```

# 评论主题建模与情感分析工具
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
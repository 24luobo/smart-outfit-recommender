# 智能穿搭推荐系统

基于多机器学习算法与LLM的个性化穿搭推荐平台，支持潮流预测与细粒度评价生成。

## 技术栈
- 后端: Python 3.8+ / Flask 3.1
- 机器学习: scikit-learn (朴素贝叶斯, K-Means, 随机森林)
- 数据库: SQLite
- 前端: HTML/CSS/JS 响应式设计
- LLM: 支持 DeepSeek/智谱GLM/通义千问/OpenAI/Ollama

## 功能模块
- 个性化穿搭推荐（多算法融合）
- 潮流趋势预测（随机森林时序模型）
- 在线试衣（AI抠图+3D展示）
- 个人衣橱管理
- AI穿搭助手

## 快速启动
!pip install -r requirements.txt
python app.py
访问 http://127.0.0.1:5000

## 项目结构
algorithms/ - ML算法实现
database/ - 数据库初始化
routes/ - Flask路由
services/ - 业务逻辑
static/ - 静态资源
templates/ - HTML模板
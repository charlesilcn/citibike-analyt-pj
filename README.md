# 项目简介
本项目对 Citi Bike 共享单车的骑行数据进行深度分析，旨在挖掘用户行为模式、时间分布特征和站点供需规律，为共享单车运营优化提供数据支持。通过数据清洗、特征工程和多维度分析，揭示会员与非会员的行为差异、早高峰热门站点的供需缺口等关键 insights。

# 项目结构
citibike-analyt-pj/
├── 01_raw_data/              # 原始数据（按年份存储）
├── 02_processed_data/        # 处理后的数据
│   ├── 2020_combined.csv     # 合并后的完整数据
│   ├── 2020_cleaned.csv      # 清洗去重后的数据
│   └── 2020_featured.csv     # 特征工程后的分析数据
├── 03_code/                  # 分析代码
│   ├── data_cleaning.py      # 数据清洗与预处理
│   ├── feature_engineering.py# 特征工程（生成时间/用户特征）
│   ├── eda_analysis.py       # 探索性数据分析（整体趋势）
│   ├── rush_hour_analysis.py # 早高峰供需缺口分析
│   └── user_route_analysis.py# 会员与非会员路线行为分析
└── 04_results/               # 分析结果
    ├── charts/               # 可视化图表（PNG格式）
    ├── user_analysis/        # 用户行为分析结果（CSV）
    └── supply_demand/        # 供需缺口分析结果（CSV）

# 核心分析内容
## 整体骑行趋势分析
月度 / 小时骑行量分布（揭示季节性和时段规律）
工作日与周末骑行差异（通勤 vs 休闲特征）
热门站点排名（出发 / 到达量 Top10）
会员与非会员行为对比
骑行路线偏好（热门起点→终点差异）
骑行距离与时长分布（会员更短 vs 非会员更长）
时间选择差异（工作日占比、时段分布）
早高峰供需缺口分析
工作日 7:00-9:00 站点供需计算（出发量 = 需求，到达量 = 供给）
缺口分类：严重缺口（需补车）、严重过剩（需挪车）
关键站点可视化（指导车辆调度）

数据准备将原始数据（CSV 格式）放入 01_raw_data/ 目录，数据需包含以下字段：started_at, ended_at, start_station_name, end_station_name, start_lat, start_lng, end_lat, end_lng, member_casual
数据处理
### 清洗数据（去重、处理缺失值）
python 03_code/data_cleaning.py

### 生成特征（时间特征、骑行时长等）
python 03_code/feature_engineering.py
执行分析
### 整体趋势分析（生成基础图表）
python 03_code/eda_analysis.py

### 会员与非会员行为分析
python 03_code/user_route_analysis.py

### 早高峰供需缺口分析
python 03_code/rush_hour_analysis.py

### 查看结果
图表保存于 04_results/charts/
详细数据结果保存于 04_results/user_analysis/ 和 04_results/supply_demand/

后续优化方向
预测模型：基于历史数据预测未来骑行需求，提前调配车辆。
POI 关联分析：结合站点周边 POI（写字楼、公园等），解释供需差异的成因。
多城市对比：扩展至其他城市的共享单车数据，分析地域差异。
交互式可视化：使用 Plotly 构建可交互仪表盘，支持动态筛选和下钻分析。

待加入：1.季度使用数据对比（季节），一日时段使用数据对比。
        2.路线聚类分析（K-means 算法实现，区分通勤 / 休闲类路线），分析会员与非会员使用偏好
        3.


技术优化类
引入并行计算（Dask/Swifter 加速大规模数据处理）
可视化升级（Plotly 交互式图表开发）
数据管道构建（Apache Airflow 实现自动化调度）
数据库集成（PostgreSQL/MongoDB 存储管理数据）

        

标题: Self Introduction
内容: 
Good morning. My name is Hailun Wang.

I graduated in 2008 with a Bachelor's degree in Software Engineering（Japanese）.
During my university studies, I also passed the CET-6 English exam.

After graduation, I worked at multinational companies and gained experience in Data Warehousing, Business Intelligence, and backend development using Python and Java.
During my career, I also took the TOEIC exam and obtained a score of 775.

After coming to Japan, I have been involved in several AI-related projects.
For example,
I used SentenceTransformer to convert knowledge base documents into vector embeddings,
stored the vectors in ChromaDB, and integrated large language models to build conversational systems that allow users to interact with the knowledge base.


标题: AI离线AI知识库框架
内容: 

标题: Agile开发（AI离线知识库开发为例子）
内容: 
标题: 软件开发流程（AI离线知识库开发为例子）
内容: 
标题: 异步，锁，多线程，多进程
内容: 




# OBIEE 项目测试流程示例（日本常见开发流程）

测试阶段：

単体テスト → 結合テスト → 総合テスト → 運用テスト

说明内容包括：
- 如何准备测试文档
- 如何准备测试数据
- 如何执行测试
- 每个阶段的成果物

------------------------------------------------------------
# 1 単体テスト（Unit Test / UT）

## 目标
验证单个组件是否正确。

OBIEE 中通常包括：
- RPD 逻辑列
- Calculation Column
- Logical SQL
- 单个 Analysis 报表


## 1.1 准备测试文档
文档名称：単体テスト仕様書

示例：
| No   | 测试项 | 条件 | 期待结果 |
|-----|------|------|------|
| UT01 | 売上金額计算 | 日期=2025/01 | 金额与DB汇总一致 |
| UT02 | 地区过滤 | Region=Tokyo | 只显示Tokyo数据 |

文档内容：
- 测试对象
- 输入条件
- 执行步骤
- 期待结果

## 1.2 准备测试数据
在数据库准备测试数据：
sales_table

date      | region | amount|
|-----|------|------|
2025-01   | Tokyo  | 100|
2025-01   | Osaka  | 200|


验证 SQL：
```sql
SELECT SUM(amount)
FROM sales_table
WHERE date='2025-01';
```

## 1.3 执行测试

步骤：

1 打开 OBIEE Analysis
2 设置过滤条件
3 执行报表
4 与 SQL 结果对比


## 1.4 成果物

| 成果物 | 内容 |
|------|------|
| 単体テスト仕様書 | 测试设计 |
| 単体テスト結果 | 执行结果 |
| SQL验证记录 | 数据验证 |
| 不具合票 | Bug记录 |



------------------------------------------------------------
# 2 結合テスト（Integration Test / IT）

## 目标
验证系统组件之间的连接是否正确。

OBIEE 系统结构：

ETL → DWH → RPD → Analysis


## 2.1 准备测试文档

文档名称：

結合テスト仕様書

示例：

| No | 测试项 | 确认内容 |
|----|------|------|
| IT01 | ETL→DWH | 数据是否正确加载 |
| IT02 | RPD映射 | 逻辑列是否正确 |
| IT03 | Report显示 | 报表是否正确显示 |


## 2.2 准备测试数据

执行 ETL Job 生成数据：

ETL Job
 ↓
DWH Table
 ↓
OBIEE Report


## 2.3 执行测试

步骤：

1 运行 ETL
2 查询 DWH 表数据
3 打开 OBIEE 报表
4 对比数据

验证逻辑：

DB数据 == 报表数据


## 2.4 成果物

| 成果物 | 内容 |
|------|------|
| 結合テスト仕様書 | 测试设计 |
| 結合テスト結果 | 执行记录 |
| 数据验证SQL | 数据校验 |
| 不具合管理表 | Bug列表 |



------------------------------------------------------------
# 3 総合テスト（System Test / ST）

## 目标
验证整个业务系统流程。

示例业务：

销售分析系统


系统流程：

ETL → DWH → OBIEE Dashboard


## 3.1 准备测试文档

文档名称：

総合テスト仕様書

示例：

| No | 业务场景 | 测试内容 |
|----|------|------|
| ST01 | 月销售报表 | 数据正确 |
| ST02 | 年度趋势分析 | 图表正确 |
| ST03 | 用户权限 | 权限控制正确 |


## 3.2 准备测试数据

一般使用：

- 接近生产环境的数据
- 历史数据

示例：

2023 sales
2024 sales
2025 sales


## 3.3 执行测试

步骤：

1 登录 OBIEE
2 打开 Dashboard
3 执行业务场景

示例：

销售经理查看区域销售

验证内容：

- 图表正确
- 数值正确
- 权限正确


## 3.4 成果物

| 成果物 | 内容 |
|------|------|
| 総合テスト仕様書 | 场景设计 |
| 総合テスト結果 | 执行记录 |
| 不具合一覧 | Bug列表 |
| 修正確認記録 | 修复确认 |



------------------------------------------------------------
# 4 運用テスト（Operation Test / OT）

## 目标
验证系统在实际运维环境中是否可以正常运行。

例如：

- 定时 ETL
- 定时报表
- 用户访问


## 4.1 准备测试文档

文档名称：

運用テスト手順書

示例：

| 项目 | 内容 |
|----|------|
| ETL定时 | 每天 01:00 |
| 报表刷新 | 每天 07:00 |
| 用户访问 | 每天 08:00 |


## 4.2 准备测试数据

使用生产模拟数据。


## 4.3 执行测试

流程：

ETL Job
 ↓
数据更新
 ↓
OBIEE Cache刷新
 ↓
用户访问报表


验证：

- Job是否成功
- 报表是否更新
- 用户是否可以正常访问


## 4.4 成果物

| 成果物 | 内容 |
|------|------|
| 運用テスト手順書 | 操作流程 |
| 運用テスト結果 | 执行记录 |
| 障害対応記録 | 故障记录 |
| 本番移行判定書 | 上线确认 |



------------------------------------------------------------
# 日本项目常见测试成果物总结

単体テスト仕様書
単体テスト結果

結合テスト仕様書
結合テスト結果

総合テスト仕様書
総合テスト結果

運用テスト手順書
運用テスト結果

不具合管理表
テスト証跡（截图）



------------------------------------------------------------
# OBIEE 项目完整测试流程

开发
 ↓
単体テスト
 ↓
結合テスト
 ↓
総合テスト
 ↓
運用テスト
 ↓
本番リリース
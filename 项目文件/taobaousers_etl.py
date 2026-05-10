import pandas as pd

file_path = r"D:\wenjian\DataAnalysis\TaobaoUserBehavior_Analysis_Project\data\UserBehavior.csv"

# 先读取前100万行
df = pd.read_csv(
    file_path,
    nrows=1000000,
    header=None,
    names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
)
print(df.head())
print(df.info())

#优化数据类型
df["behavior_type"] = df["behavior_type"].astype("category")
df["user_id"] = df["user_id"].astype("int32")
df["item_id"] = df["item_id"].astype("int32")
df["category_id"] = df["category_id"].astype("int32")

#时间转换
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday

#用户行为结构分析
#行为数量统计
behavior_count = df["behavior_type"].value_counts()
print("行为数量统计：")
print(behavior_count)
#行为占比统计
behavior_ratio = df["behavior_type"].value_counts(normalize=True)
print("行为占比：")
print(behavior_ratio)

#转化漏斗分析
# 不同行为的用户数
pv_users = df[df["behavior_type"] == "pv"]["user_id"].nunique()
cart_users = df[df["behavior_type"] == "cart"]["user_id"].nunique()
buy_users = df[df["behavior_type"] == "buy"]["user_id"].nunique()
print("浏览用户数:", pv_users)
print("加购用户数:", cart_users)
print("购买用户数:", buy_users)
#计算转化率
cart_rate = cart_users / pv_users
buy_rate = buy_users / pv_users
cart_to_buy_rate = buy_users / cart_users
print("加购率:", cart_rate)
print("购买率:", buy_rate)
print("加购→购买转化率:", cart_to_buy_rate)

#时间维度分析
#分析购买高峰小时
df_buy = df[df["behavior_type"] == "buy"]
hour_analysis = df_buy.groupby("hour").size().sort_values(ascending=False)
print("购买高峰小时：")
print(hour_analysis)
#分析周几购买最多
weekday_analysis = df_buy.groupby("weekday").size().sort_values(ascending=False)
print("周几购买最多：")
print(weekday_analysis)

#爆款商品分析
top_items = df_buy.groupby("item_id").size().sort_values(ascending=False).head(20)
print("销量Top20商品：")
print(top_items)

#商品转化率分析
# 浏览商品用户数
pv_item = df[df["behavior_type"] == "pv"].groupby("item_id")["user_id"].nunique()
# 购买商品用户数
buy_item = df[df["behavior_type"] == "buy"].groupby("item_id")["user_id"].nunique()
# 合并
item_conversion = pd.concat([pv_item, buy_item], axis=1)
item_conversion.columns = ["pv_users", "buy_users"]
# 删除没有浏览的商品
item_conversion = item_conversion.dropna()
# 只保留浏览人数>=5的商品（防止极端值）
item_conversion = item_conversion[item_conversion["pv_users"] >= 5]
#只分析：同时出现在 pv 和 buy 里的商品并且：buy_users <= pv_users
item_conversion = item_conversion[item_conversion["buy_users"] <= item_conversion["pv_users"]]
# 计算转化率
item_conversion["conversion_rate"] = item_conversion["buy_users"] / item_conversion["pv_users"]
# 排序
top_conversion = item_conversion.sort_values("conversion_rate", ascending=False).head(20)
print(top_conversion)

#用户价值分析（RFM模型），由于数据集没有金额，所以只能R+F
#准备购买数据
df_buy = df[df["behavior_type"] == "buy"].copy()
#计算最近购买时间（R）
last_purchase = df_buy.groupby("user_id")["datetime"].max().reset_index()
last_purchase.columns = ["user_id", "last_purchase_time"]
# 数据集中的最大时间
max_date = df["datetime"].max()
# 计算距离最近购买天数
last_purchase["recency"] = (max_date - last_purchase["last_purchase_time"]).dt.days
#计算每个用户购买次数（F）
frequency = df_buy.groupby("user_id").size().reset_index()
frequency.columns = ["user_id", "frequency"]
#合并 R 和 F
rfm = pd.merge(frequency, last_purchase[["user_id", "recency"]], on="user_id")
print(rfm.head())
#做简单分层
f_median = rfm["frequency"].median()
r_median = rfm["recency"].median()
rfm["F_score"] = rfm["frequency"].apply(lambda x: 1 if x > f_median else 0)
rfm["R_score"] = rfm["recency"].apply(lambda x: 1 if x <= r_median else 0)
rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str)
#统计用户分布
rfm_distribution = rfm["RFM_score"].value_counts()
print("用户分层分布：")
print(rfm_distribution)

#导出 RFM 表
rfm.to_csv("rfm_table.csv", index=False)
#导出商品转化率表
item_conversion.to_csv("item_conversion.csv")
#导出每日购买汇总
daily_buy = df_buy.groupby("date").size().reset_index()
daily_buy.columns = ["date", "buy_count"]
daily_buy.to_csv("daily_buy.csv", index=False)


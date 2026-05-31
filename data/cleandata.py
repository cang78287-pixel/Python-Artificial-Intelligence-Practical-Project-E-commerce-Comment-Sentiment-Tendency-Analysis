import pandas as pd
import jieba
import re

# 1. 加载数据
df = pd.read_csv("chnsenticorp_raw.csv")

# 2. 定义停用词（这里简单列举，实际项目中可以下载专门的停用词表）
stop_words = set(
    ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要',
     '去', '你', '会', '着', '没有', '看', '好', '自己', '这','就是', '觉得', '比较', '这种', '一个', '还是', '可以', # 语气词
    '评论', '购买', '卖家', '京东', '当当网', '淘宝','我们','感觉','酒店', '房间', '本书', '书', '电脑', '笔记本', '携程', '卖家', '京东',

    '但是', '什么', '结果', '而且', '就是', '虽然', '觉得', '感觉', '还是',
    '比较', '有点', '一个', '这种', '因为', '所以', '如果', '那么', '这样',
    '内容', '作者', '总体', '评价', '评论', '东西', '地方', '时候', '问题'])



def clean_text(text):
    if not isinstance(text, str):
        return ""

    # A. 去除特殊符号、表情、网址
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # B. 使用 jieba 分词
    words = jieba.lcut(text)

    # C. 去停用词
    words = [w for w in words if w not in stop_words and len(w) > 1]

    return " ".join(words)


print("正在进行数据清洗和分词，请稍候...")
df['clean_text'] = df['text'].apply(clean_text)

# 3. 过滤掉清洗后变为空的评论
df = df[df['clean_text'] != ""]

# 4. 保存为适配实验的数据集
df[['label', 'clean_text']].to_csv("chnsenticorp_cleaned.csv", index=False, encoding="utf_8_sig")
print("清洗完成！已生成：chnsenticorp_cleaned.csv")
print(df[['label', 'clean_text']].head())

# 查看总行数和总列数
print(f"清洗后的数据集规模: {df.shape}")

# 查看正负面标签的分布情况（确认数据是否均衡）
print("情感标签分布情况:")
print(df['label'].value_counts())

# 1. 加载你最初获取的原始数据
df = pd.read_csv("chnsenticorp_raw.csv")

# 2. 简单的行业分类逻辑（根据数据特征进行切分）
# 提示：ChnSentiCorp 原始数据通常前几千条是酒店，中间是书籍，最后是电脑
# 或者你可以根据文本中是否包含关键词来切分
hotel_df = df[df['text'].str.contains('酒店|房间|宾馆|携程', na=False)]
book_df = df[df['text'].str.contains('书|作者|内容|阅读', na=False)]
nb_df = df[df['text'].str.contains('电脑|笔记本|键盘|散热', na=False)]

print(f"切分完成：酒店({len(hotel_df)}条), 书籍({len(book_df)}条), 电脑({len(nb_df)}条)")

# 3. 分别保存，交给下一步的可视化
hotel_df.to_csv("data_hotel.csv", index=False)
book_df.to_csv("data_book.csv", index=False)
nb_df.to_csv("data_notebook.csv", index=False)
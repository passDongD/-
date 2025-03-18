import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pandas as pd
questions =json.load(open("questions.json",encoding='utf-8'))

pdf=pdfplumber.open("初赛训练数据集.pdf")
len(pdf.pages)
pdf.pages[0].extract_text()
pdf_content=[]
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page'+str(page_idx+1),
        'content': pdf.pages[page_idx].extract_text()
    })
df = pd.DataFrame(pdf_content)

df.to_excel("初赛训练数据集.xlsx", index=False)
# question_words=[''.join(jieba.lcut(x['question']))for x in questions]
# pdf_content_words =[''.join(jieba.lcut(x['content']))for x in pdf_content]
# # 提取TFIDF
# tfidf = TfidfVectorizer()
# tfidf.fit(question_words + pdf_content_words)
# question_feat =tfidf.transform(question_words)# 转换为矩阵
# pdf_content_feat =tfidf.transform(pdf_content_words)
# question_feat =normalize(question_feat)#归一化
# pdf_content_feat =normalize(pdf_content_feat)
# # 通过TFIDF进行检索
# for query_idx,feat in enumerate(question_feat):
#     score= feat @ pdf_content_feat.T
#     score =score.toarray()[0]
#     max_score_page_idx=score.argsort()[::-1][0]+ 1
#     questions[query_idx]['reference']='page_'+ str(max_score_page_idx)
# with open('submit_tfidf_retrieval_top1.json', 'w', encoding='utf8') as up:
#     json.dump(questions,up,ensure_ascii=False, indent=4)

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import time
from loguru import logger as lg
import faiss
import math
import torch
import json
#### load model
lg.info("load model")
model_name = "aspire/acge_text_embedding"  # "bert-base-chinese"  #'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
lg.info("cuda device i %s" % (str(model.device)))
embedding_size = 1792
#### load data and clean it
lg.info("load data and clean it")
# data_path = os.path.join(os.path.realpath("."), "stanard_data", "深航数据.xlsx")
# data_path
#
# data = pd.read_excel(data_path)
#
# data.columns = ["DE Number", "type", "created_date", "knowledge_question", "plan_measure", "measure", "measure_num"]
#
# data["DE Number"].nunique()
#
# data[data.duplicated(subset=["DE Number"], keep=False)].shape
#
# data[data.duplicated(keep=False)].shape
#
# data[data.duplicated(subset=["DE Number"], keep=False)].sort_values(by=["DE Number"], ascending=False).head()
#
# data.drop_duplicates(subset=["DE Number"], keep="first", inplace=True)
# data = data[~data["knowledge_question"].isna()].copy()
# data.reset_index(inplace=True)
#
# #### create a faiss index for sub_knowledge_base
# lg.info("create a faiss index for sub_knowledge_base")
# sub_data = data[data["type"] == "73N"].copy()  # data.head(2000)
sub_data = pd.read_excel(os.path.join(os.path.realpath("."),"knowledge","初赛训练数据集.xlsx"))
lg.info("data shape is %s,%s" % sub_data.shape)


def check_dataframe_valid(df):
    if not isinstance(df, pd.DataFrame):
        raise Exception('df should be pandas.DataFrame')
    return


def create_faiss_index(data, embedding_size=384, n_clusters=10, nprobe=5):
    start_time = time.time()
    check_dataframe_valid(data)

    n_clusters = int(math.sqrt(len(data)))
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe
    lg.info("Encode the corpus. This might take a while")
    corpus_sentences = list()
    for i, row in data.iterrows():
        corpus_sentences.append(row['content'])
    corpus_sentences = list(corpus_sentences)

    corpus_sentences = [str(sentence) for sentence in corpus_sentences]
    # 记录语料库中的条目类型，确保其正确性
    lg.info("语料库句子类型：%s", {type(sent).__name__ for sent in corpus_sentences})

    corpus_embeddings = model.encode(corpus_sentences, convert_to_numpy=True, normalize_embeddings=True)
    embs = [np.array(i) for i in corpus_embeddings]
    lg.info("finish encoding for knowledge_question, time cost " + str(time.time() - start_time))
    # Create the FAISS index
    lg.info("Start creating FAISS index")
    # First, we need to normalize vectors to unit length
    try:
        # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)
        # Finally we add all embeddings to the index
        index.add(corpus_embeddings)
    except Exception as e:
        lg.debug(e)
    lg.info("finish creating FAISS index, time cost " + str(time.time() - start_time))
    return index, embs


# knowledge_index, embs = create_faiss_index(sub_data, embedding_size=embedding_size, n_clusters=10, nprobe=5)
# sub_data["embs"] = embs
# sub_data.to_excel(os.path.join(os.path.realpath("."), "knowledge", "初赛训练数据集.xlsx"))


####
# lg.info("Start saving index")
def serialize_faiss_index(index, directory_path: str, file_name: str) -> bool:
    """
    Serialize a Faiss index and write it to Amazon EFS.

    :param index: Faiss index object.
    :param directory_path: the directory path where save index file.
    :param file_name: Name of the file to write.
    :return: True if writing is successful, False if writing fails.
    """
    try:
        # Specify the file path to write to
        file_path = os.path.join(directory_path, file_name)
        faiss.write_index(index, file_path)
        lg.info(f"Faiss index successfully written to {file_path}")
        return True
    except Exception as e:
        lg.info(f"Error writing Faiss index: {str(e)}")
        return False


# index_dir = os.path.join(os.path.realpath('.'), 'index')
# lg.info("save test index into %s" % index_dir)
index_file_name = "car_acge_index"
# serialize_faiss_index(knowledge_index, index_dir, index_file_name)


### load index
def deserialize_faiss_index(directory_path: str, file_name: str):
    """
    Read a Faiss index and deserialize it.

    :param directory_path:  the directory path where save index file.
    :param file_name: Name of the file to read.
    :return: Faiss index object or None if reading fails.
    """
    try:
        # Specify the file path to read from
        file_path = os.path.join(directory_path, file_name)
        # Deserialize the Faiss index
        index = faiss.read_index(file_path)
        return index
    except Exception as e:
        print(f"Error reading Faiss index: {str(e)}")
        return None


lg.info("load_index")
index_dir = os.path.join(os.path.realpath('.'),'index')
knowledge_index = deserialize_faiss_index(index_dir,index_file_name)

#### Retreive top5 answer from knowledge data for input query
lg.info("Retreive top5 answer from knowledge data for input query")

query = "前排座椅通风”的相关内容在第几页？"


def compare(question, knowledge_index, knowledge_data, top_k_hits=5, embedding_size=384):
    question_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    question_embedding = np.array(question_embedding).reshape(1, embedding_size)
    # question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1)[:, None]
    if knowledge_index is None:
        lg.info("There's no knowledge_index")
        return None
    distances, corpus_ids = knowledge_index.search(question_embedding, top_k_hits)
    response = []
    for i in range(len(corpus_ids)):
        hits = [{'corpus_id': corpus_id, 'score': score} for corpus_id, score in zip(corpus_ids[i], distances[i])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        for hit in hits[0:top_k_hits]:
            row = knowledge_data.iloc[hit['corpus_id'],]
            index = str(row["page"])

            knowledge_question = str(row["content"])


            sim = int(hit['score'] * 1000) / 10.0
            response_dict = {'question': question,
                             'index': index,
                             'knowledge_question': knowledge_question,
                             'score': sim,
                             'algorithm': 'FAISS-KnowledgeQuestion'
                             }
            response.append(response_dict)
    final_response = {"FAISS-KnowledgeQuestion": response}
    return final_response


knowledge_data = sub_data
response = compare(query, knowledge_index, knowledge_data, top_k_hits=10, embedding_size=embedding_size)
print(response)


questions =json.load(open("questions_demo.json",encoding='utf-8'))
for query_idx, question in enumerate(questions):
    query = question['question']
    response = compare(query, knowledge_index, knowledge_data, top_k_hits=10, embedding_size=embedding_size)
    # 假设 response 包含了一个 page index 或其他标识符
    questions[query_idx]['reference'] = str(response['FAISS-KnowledgeQuestion'])
    print(question)
#将更新的数据保存回 JSON 文件
# with open("questions_demo.json", "w", encoding='utf-8') as file:
#     json.dump(questions, file, ensure_ascii=False, indent=4)
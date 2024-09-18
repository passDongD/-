from langchain_core.output_parsers import StrOutputParser

from langchain_rag import *
import torch
from loguru import logger as lg
import time
import os
import pandas as pd
import pickle
from typing import (Any)
from rag_main import rag_main
import json
# from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# lg.info("start load model")
# embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
# model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
# embedding_model = load_hf_embedding_model(embedding_model_name,model_kwargs)
# lg.info("end load model")
# f3n_data_path = "C:\\Users\\admin\\Desktop\\shenhang\\knowledge\\test_f3n_data.csv"
# f3n_data_path = os.path.join(os.path.realpath('.'),'..','knowledge','test_f3n_data.csv')
# f3n_data = pd.read_csv(f3n_data_path)


# def run_create_langchain_faiss_vectorstore_main():
### generate create_langchain_faiss_vectorstore_main
# embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
# #embedding_model_name = "/Users/libo/.cache/torch/sentence_transformers/aspire_acge_text_embedding"
# embedding_model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
# f3n_data_path = "C:\\Users\\admin\\Desktop\\shenhang\\knowledge\\test_f3n_data.csv"
# #f3n_data_path = os.path.join(os.path.realpath('.'),'..','knowledge','f3n_data.csv')
# do_you_have_an_indexed_meta_faiss = True
# faiss_vectorstore_folder_path = "C:\\Users\\admin\\Desktop\\shenhang\\langchain_vectorstore"
# faiss_vectorstore_file_name = "langchain_faiss_f3n_vstore"
# meta_faiss_index_dir = "C:\\Users\\admin\\Desktop\\shenhang\\index"
# meta_faiss_index_file_name = "test_acge_index"
# create_langchain_faiss_vectorstore(embedding_model_name,embedding_model_kwargs,f3n_data_path,do_you_have_an_indexed_meta_faiss,meta_faiss_index_dir,meta_faiss_index_file_name,faiss_vectorstore_folder_path,faiss_vectorstore_file_name)

# initialize the bm25 retriever and faiss retriever
bm25_retriever = load_retriever(
    os.path.join("D:\\python workspace\\shenhang-test_env\\ai\\", 'langchain_retriever', 'bm25_retriever_car'))
faiss_retriever = load_retriever(
    os.path.join("D:\\python workspace\\shenhang-test_env\\ai\\", 'langchain_retriever', 'faiss_retriever_car'))
keyword_data = pd.read_csv(os.path.join("D:\\python workspace\\shenhang-test_env\\ai\\", 'knowledge', '初赛训练数据集.csv'))
historical_data = pd.read_csv(os.path.join("D:\\python workspace\\shenhang-test_env\\ai\\", 'knowledge', '初赛训练数据集.csv'))


@lg.catch
def customize_rerank(query: str) -> pd.DataFrame:
    """ Rerank the faiss result base on using bm25 retrieve result. bm25 will match the keyword of 4bit chapter with input question.
    Args: query: input query
    Returns: return faiss result sorted by bm25
    """
    lg.info("bm25 retrieve")
    bm25_retriever_result = bm25_retriever.invoke(query)
    bm25_retriever_result_row_ls = [doc.metadata["row"] for doc in bm25_retriever_result]
    bm25_retriever_keyword_data = keyword_data.loc[bm25_retriever_result_row_ls, :]
    lg.info(bm25_retriever_keyword_data["index"])
    lg.info("fasiss retrieve")
    faiss_retrieve_result = faiss_retriever.invoke(query)
    faiss_retriever_result_row_ls = [doc.metadata["row"] for doc in faiss_retrieve_result]
    faiss_retriever_historical_data = historical_data.loc[faiss_retriever_result_row_ls, :]
    lg.info("check if there're multiple 4bit measure num in bm25 result")
    if len(np.unique(bm25_retriever_keyword_data["index"])) != bm25_retriever_keyword_data.shape[0]:
        lg.info("yes,there're multiple 4bit measure num")
        bm25_retriever_4bit_num_sort_values = bm25_retriever_keyword_data.value_counts(subset="index")
        bm25_retriever_keyword_4bit_num = bm25_retriever_4bit_num_sort_values.index.to_list()

        faiss_retriever_historical_data_in_keyword = faiss_retriever_historical_data[
            faiss_retriever_historical_data["4_bit_measure_num"].isin(bm25_retriever_keyword_4bit_num)].copy()
        faiss_retriever_historical_data_notin_keyword = faiss_retriever_historical_data[
            ~faiss_retriever_historical_data["4_bit_measure_num"].isin(bm25_retriever_keyword_4bit_num)].copy()

        if not faiss_retriever_historical_data_in_keyword.empty:
            lg.info("4bit measure num also in faiss result")
            # Create a dictionary mapping integers to their positions in the custom order
            order_mapping = {val: i for i, val in enumerate(bm25_retriever_keyword_4bit_num)}

            # Create a temporary column for sorting based on the custom order
            faiss_retriever_historical_data_in_keyword.loc[:, 'sort_temp'] = faiss_retriever_historical_data_in_keyword[
                '4_bit_measure_num'].map(order_mapping).copy()

            # Sort the DataFrame by the temporary column
            faiss_retriever_historical_data_in_keyword_sorted = faiss_retriever_historical_data_in_keyword.sort_values(
                'sort_temp').copy()
            faiss_retriever_historical_data_in_keyword_sorted.drop(columns=['sort_temp'], inplace=True)

            faiss_retriever_historical_data = pd.concat(
                [faiss_retriever_historical_data_in_keyword_sorted, faiss_retriever_historical_data_notin_keyword])
    lg.info("customize_rerank done")
    return faiss_retriever_historical_data

# def generate_query(query):
#     client = ZhipuAI(api_key="5206a3dca0ba2755a51b3401f7417fcb.yqd5rvnm19Xm8ZoS")  # 填写您自己的APIKey
#     content = "简介的回答用户问题。问题：{" + str(query) + "}"
#     print(content)
#     response = client.chat.completions.create(
#         model="glm-4",  # 填写需要调用的模型名称
#         messages=[
#             {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
#             {
#                 "role": "user",
#                 "content": content
#             },
#         ],
#         stream=False,
#     )
#
#     # print(response)
#     print(response | StrOutputParser() | (lambda x: x.split("\n")))
#     return str(response | StrOutputParser() | (lambda x: x.split("\n")))


def post_process(faiss_retriever_historical_data: pd.DataFrame, top_k: Optional[int] = 5) -> Dict[str, List[str]]:
    """Post process the result of sorted faiss result
    Args:
        faiss_retriever_historical_data: the result of sorted faiss result
        top_k: return topk
    Return:
    """
    final_result = {}

    historical_measure_num_ls = []
    historical_measure_ls = []
    n = 0
    for idx, row_data in faiss_retriever_historical_data.iterrows():
        n += 1
        if n > top_k:
            break

        historical_measure_num_ls.append(row_data["index"])
        historical_measure_ls.append(row_data["content"])

    final_result["content"] = historical_measure_ls
    final_result["page"] = historical_measure_num_ls
    return final_result


# def ircot(query):
#     client = ZhipuAI(api_key="80455c76392763b39b00bd3bb978f153.CLwhcgyIPKXPBLAE")  # 填写您自己的APIKey
#     content = "你是一个能根据输入问题生成多个子问题的有用助手。目标是将输入分解为一组可以独立回答的子问题。生成与原始问题相关的 3 个子问题。原始问题：{" + str(query) + "}。输出 3 个可以独立回答的子问题："
#     print(content)
#     response = client.chat.completions.create(
#         model="glm-4",  # 填写需要调用的模型名称
#         messages=[
#             {"role": "system", "content": "你是一个能根据输入问题生成多个子问题的有用助手。目标是将输入分解为一组可以独立回答的子问题"},
#             {
#                 "role": "user",
#                 "content": content
#             },
#         ],
#         stream=False,
#     )
#     print(response.choices[0])
#     # print(response | StrOutputParser() | (lambda x: x.split("\n")))
#     return str(response.choices[0])

@lg.catch
def rag_main(query: str, topk) -> Dict[str, List[str]]:
    start_time = time.time()
    lg.info(f"rag_main: {query}")

    lg.info("start answer question")
    faiss_retriever_historical_data = customize_rerank(query)
    lg.info("start post proces")
    final_result = post_process(faiss_retriever_historical_data, topk)
    lg.info("end answer question")

    end_time = time.time()
    lg.info(f"rag_main costing time: {end_time - start_time}")
    return final_result
import json
from zhipuai import ZhipuAI
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
client = ZhipuAI(api_key="3a6d6d1fdb1e57ab1892546a6503e1ff.TRZZqHykpiXvmQn7") # 填写您自己的APIKey

def process_queries(queries):
    for query_obj in queries:

        reference = rag_main(query_obj["question"], 1)

        content = "根据我给出的问题内容返回一个详细的中文回答。必须使用我提供的资料来回答，不许自己给出解释!!!如果你不知道答案，就说你不知道，不要试图编造答案。给出的内容包括汽车手册的相关内容。其中手册的内容上下文中可能有多个相关信息，你需要自己去识别对应的内容。请从上下文中选择相关的文本来回答我的输入问题，不相关的内容请忽略。prompt =开始吧!我的输入:{" + str(
            query_obj["question"]) + "}，给出的上下文:{" + str(reference['content']) + "}"
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "system",
                 "content": "你是一个汽车专家，你擅长编写和回答汽车相关的用户提问，帮我结合给定的资料，回答下面的问题如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。如果问题可以从资料中获得，则请逐步回答。以下是回答的两种示例：question:“前排座椅通风”的相关内容在第几页？answer：前排座椅通风的相关内容在第115页和第117页。  question:尾门自动打开/关闭的机制是如何感应障碍物的？answer：结合给定的资料，无法回答问题。"},
                {
                    "role": "user",
                    "content": content
                },
            ],
        )

        text = response.choices[0].message.content
        query_obj["answer"] = text

        query_obj["reference"] = reference['page'][0]
        print(query_obj)
    return queries

if __name__ == "__main__":
    ###
    #query = "关于车辆的儿童安全座椅固定装置，在哪一页可以找到相关内容？"
    #top_k = 1
    # query = generate_query(query)
    # query = ircot(query)
    # final_result = rag_main(query, top_k)
    # # final_result = rag_main(query)
    # print(final_result)

    file_path = 'questions_demo92.json'
    queries = load_json(file_path)
    updated_queries = process_queries(queries)
    save_json(updated_queries, file_path)




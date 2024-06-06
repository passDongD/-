from langchain_rag import *
import torch
from loguru import logger as lg
import time
import os
from faiss_indexer import RagIndexer
import pandas as pd

lg.info("start load model")
embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
embedding_model = load_hf_embedding_model(embedding_model_name,model_kwargs)
lg.info("end load model")
# f3n_data_path = "C:\\Users\\admin\\Desktop\\shenhang\\knowledge\\test_f3n_data.csv"
# f3n_data = pd.read_csv(f3n_data_path)
sub_data = pd.read_excel(os.path.join(os.path.realpath("."),"knowledge","初赛训练数据集.xlsx"))


def remove_bom_and_save(file_path, new_file_path):
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 去除BOM
        if content.startswith('\ufeff'):
            content = content[1:]

    # 保存新的内容到文件
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(content)



def create_langchain_faiss_main(embedding_model_name, model_kwargs,  source_column, is_load_meta_faiss,
                                knowledge_index, knowledge_docstore, knowledge_index_to_docstore_id,
                                faiss_vectorstore_folder_path, faiss_vectorstore_file_name):
    start_time = time.time()
    lg.info("start load model")
    embedding_model = load_hf_embedding_model(embedding_model_name, model_kwargs)
    lg.info("end load model")

    lg.info("start load data")
    data_path = os.path.join(os.path.realpath("."), "knowledge", "初赛训练数据集_new.csv")


    custom_docs = load_csv_into_langchain(data_path,encoding_type="utf-8", source_column ="index")
    lg.info("end load data")

    lg.info("start create langchain faiss")

    if is_load_meta_faiss:
        langchain_faiss = load_meta_faiss_into_langchain(embedding_model, knowledge_index,
                                                         customize_docstore=knowledge_docstore,
                                                         customize_index_to_docstore_id=knowledge_index_to_docstore_id)
        faiss_as_retrieve

    else:
        faiss_vectorstore = FAISS.from_documents(
            custom_docs, embedding_model
        )
    lg.info("save_faiss_vectorstore")
    save_langchain_faiss_vectorstore(faiss_vectorstore, faiss_vectorstore_folder_path, faiss_vectorstore_file_name)
    lg.info("end save_faiss_vectorstore")
    end_time = time.time()
    lg.info(f"create_langchain_faiss_main costing time: {end_time - start_time}")
    return


def post_process(docs):
    final_result = {}
    historical_question_ls = []
    historical_measure_num_ls = []

    for doc in docs:

        doc_source = doc.metadata["source"]

        historical_question_ls.append(doc)
        historical_measure_num_ls.append(doc_source)


    final_result["content"] = historical_question_ls
    final_result["page"] = historical_measure_num_ls
    return final_result


def rag_main(query: str):
    start_time = time.time()
    lg.info(f"rag_main: {query}")

    # lg.info("start load data")
    # data_path = os.path.join(os.path.realpath("."),"knowledge","初赛训练数据集_new.csv")
    # source_column = "index"
    # custom_docs = load_csv_into_langchain(data_path,source_column=source_column)
    # lg.info("end load data")
    #
    # lg.info("start init bm25_retriever")
    # bm25_retriever = initialize_bm25_retriever(custom_docs)
    # bm25_retriver_path = os.path.join(os.path.realpath('.'),'langchain_retriever','bm25_retriever.pkl')
    # bm25_retriever = load_retriever(bm25_retriver_path)
    # lg.info("end init bm25_retriever")
    #
    # lg.info("init langchain_faiss_vectorstore")
    # faiss_vectorstore_folder_path = os.path.join(os.path.realpath('.'),'..','langchain_faiss_vector_store')
    # faiss_vectorstore_file_name ="langchain_faiss_index_car"
    # langchain_faiss_vectorstore = load_langchain_faiss_vectorstore(embedding_model,faiss_vectorstore_folder_path,faiss_vectorstore_file_name,allow_dangerous_deserialization=True)
    # lg.info("end langchain_faiss_vectorstore")
    #
    # search_kwargs={"k": 3}
    # langchain_faiss_retriever = faiss_as_retrieve(langchain_faiss_vectorstore,search_kwargs)
    #
    lg.info("init ensemble_retriever")
    # weights = [0.5,0.5]
    # ensemble_retriever = init_ensemble_retriever(bm25_retriever,langchain_faiss_retriever,weights)
    ensemble_retriver_path = "C:\\Users\\admin\\Desktop\\shenhang\\langchain_retriever\\ensemble_retriever_car.pkl"  # os.path.join(os.path.realpath('.'),'..','langchain_retriever','ensemble_retriever.pkl')
    # save_retriever(ensemble_retriever,ensemble_retriver_path)
    ensemble_retriever = load_retriever(ensemble_retriver_path)
    lg.info("end ensemble_retriever")

    lg.info("start answer question")
    docs = ensemble_retriever.invoke(query)
    final_result = post_process(docs)
    # source_ls = [docs[0].metadata["source"] for doc in docs]

    lg.info("end answer question")

    end_time = time.time()
    lg.info(f"rag_main costing time: {end_time - start_time}")
    return final_result


if __name__ == "__main__":
    # file_path = os.path.join(os.path.realpath("."), "knowledge", "初赛训练数据集.csv")
    # new_file_path = os.path.join(os.path.realpath("."), "knowledge", "初赛训练数据集_new.csv")  # 可以是相同的文件名来覆盖原文件
    # remove_bom_and_save(file_path, new_file_path)
    # print(1)
    # embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
    # model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    # source_column = "index"
    # rag_index = RagIndexer()
    # index_dir = os.path.join(os.path.realpath('.'),'index')
    # index_file_name = "car_acge_index"
    # knowledge_index = rag_index.deserialize_faiss_index(index_dir,index_file_name)
    # knowledge_docstore = None
    # knowledge_index_to_docstore_id = None
    # is_load_meta_faiss =False
    # faiss_vectorstore_folder_path = os.path.join(os.path.realpath('.'), '..', 'langchain_faiss_vector_store')
    # faiss_vectorstore_file_name = "langchain_faiss_index_car"
    # create_langchain_faiss_main(embedding_model_name, model_kwargs, source_column, is_load_meta_faiss,
    #                             knowledge_index, knowledge_docstore, knowledge_index_to_docstore_id,
    #                             faiss_vectorstore_folder_path, faiss_vectorstore_file_name)

    query = "关于车辆的儿童安全座椅固定装置，在哪一页可以找到相关内容？"
    final_result = rag_main(query)
    print(final_result)

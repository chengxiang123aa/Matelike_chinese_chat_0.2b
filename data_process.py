import ujson
import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import pandas as pd 
import numpy as np
import time
from fastparquet import ParquetFile, write
import codecs, csv
from rich import progress   
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："


def delete_file(file: str)-> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1
    return ans

def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True 

    write(file_name, data_frame, compression='GZIP',append=append)

def read_and_write_template(read_file: str, write_to_file: str, call_back: object, group_cnt: int=10000) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    group_cnt: parquet file分割行数

    '''

    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    
    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []
        append = cur_rows.append
        for line in f_read:
            try:
                raw_line_cnt += 1
                write_dict = call_back(line)
                if write_dict is None: continue
                keep_line_cnt += 1
                append(write_dict)

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append

            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e
        
        # end for
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []
    
    end = time.time()

    log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'.format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start))

def process_belle_knowledge_sft_dataset(response_less_words: int=15, group_cnt: int=10000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
        'Belle_open_source_0.5M.json',
        'generated_chat_0.4M.json',
        'train_2M_CN.json',
    ]

    save_file = 'full_sft_dataset_3m_based_belle.parquet'
    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)
    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if '翻译' in prompt or 'translate' in prompt.lower():
            return None
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(response) < response_less_words:
            return None
        prompt = remove_duplicate_punctuation(prompt)
        response = remove_duplicate_punctuation(response)
        if len(response) < response_less_words:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }
        return write_dict

    for file in file_names:
        read_and_write_template(file, save_file, process_function)

process_belle_knowledge_sft_dataset()
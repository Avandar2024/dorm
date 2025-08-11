import json
import tomllib
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from itertools import cycle

import leidenalg as la
import numpy as np
import pandas as pd
import tenacity
from igraph import Graph
from openai import OpenAI

from prompt import prompts_message

Partition = la.ModularityVertexPartition


def is_int(string: str) -> bool:
    """检查字符串是否可以转换为整数（正负数、前导空格都可以）"""
    try:
        int(string)
        return True
    except ValueError:
        return False


def create_clients() -> list[OpenAI]:
    with open('api_key.toml', 'rb') as f:
        config = tomllib.load(f)
    api_key: list = config['qwen_api_key']
    clients = [
        OpenAI(api_key=key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
        for key in api_key
    ]
    return clients


def chunk_dict(data: dict, chunk_size: int) -> Generator[dict, None, None]:
    """将字典分割成多个小字典"""
    items = list(data.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i : i + chunk_size])


def process_chunk(chunk: dict, client: OpenAI) -> dict:
    """处理单个数据块"""
    print(f'Processing chunk with {len(chunk)} entries...')
    one_chat = partial(
        client.chat.completions.create,
        model='qwen-plus',
        response_format={'type': 'json_object'},
        extra_body={'enable_thinking': False},
    )

    try:
        response = one_chat(messages=prompts_message(chunk))
        json_output = response.choices[0].message.content
        # print(f'LLM response: {json_output}')
        if not json_output or json_output == 'null':
            raise ValueError('LLM response is empty')
        result: dict = json.loads(json_output)
        for key, value in result.items():
            value: list
            if not is_int(key):
                raise ValueError(f'Invalid key in LLM response: {key}')
            if not all(is_int(item) for item in value):
                raise ValueError(f'Invalid value in LLM response: {value}')
        return json.loads(json_output)  # type: ignore
    except Exception as e:
        print(f'Error processing chunk: {e}')
        raise


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def llm_process(data: dict) -> dict:
    """使用LLM对数据进行分类，使用并发和客户端轮换"""
    print(f'Processing data with {len(data)} entries using LLM...')
    chunks = list(chunk_dict(data, chunk_size=100))
    clients = create_clients()

    # 如果没有客户端可用，抛出错误
    if not clients:
        raise ValueError('No API clients available')

    # 创建一个客户端轮询器
    client_cycle = cycle(clients)

    # 创建工作队列
    tasks = []
    with ThreadPoolExecutor(max_workers=min(len(clients), 5)) as executor:
        # 提交所有任务，每个任务使用下一个可用的客户端
        for chunk in chunks:
            client = next(client_cycle)  # 轮换获取下一个客户端
            task = executor.submit(process_chunk, chunk, client)
            tasks.append(task)

        # 收集结果
        results = {}
        for future in as_completed(tasks):
            try:
                chunk_result = future.result()
                # print(f'Chunk processed successfully: {chunk_result}')
                results.update(chunk_result)
            except Exception as e:
                print(f'Task failed with error: {e}')

    # 检查结果是否为空
    if not results:
        raise ValueError('All LLM requests failed or returned empty responses')

    # 保存结果到文件
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def community_sort(g: Graph, communities: list) -> Partition:
    """对社区进行排序"""
    num_communities = len(list(set(communities)))
    adjencency_matrix = np.zeros((num_communities, num_communities), dtype=int)
    for edge in g.es:
        source_community = communities[edge.source]
        target_community = communities[edge.target]
        if source_community != target_community:
            adjencency_matrix[source_community][target_community] += 1
            adjencency_matrix[target_community][source_community] += 1
    total_conn = np.sum(adjencency_matrix, axis=1)
    sorted_indices = np.argsort(total_conn)[::-1]
    old_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_indices)}
    new_membership = [int(old_to_new_id_map[old_id]) for old_id in communities]
    new_partition = Partition(g, new_membership)
    return new_partition


def community_find(group: dict) -> pd.DataFrame:
    """解析llm的输出"""
    group_info = {str(k): [str(item) for item in v] for k, v in group.items()}
    # print(group_info)
    # 生成无向图
    g: Graph = Graph.ListDict(group_info, directed=False)
    # leidenalg社区发现
    partition = la.find_partition(g, Partition)
    sorted_partition = community_sort(g, partition.membership)
    result = pd.DataFrame({'group': sorted_partition.membership})
    # print(result['group'])
    return result


def append_ai_col(dst: pd.DataFrame):
    """给DataFrame添加ai列"""
    ai_col: pd.Series = dst.iloc[:, -1]  # ai列是最后一列
    dst.drop(columns=dst.columns[-1], inplace=True)
    group: dict = llm_process(ai_col.to_dict())
    # 解析JSON字符串
    df = community_find(group)
    # 将结果添加到原DataFrame中
    dst['26_ai_group'] = df['group']


if __name__ == '__main__':
    # test
    # raw = '{"a": ["b", "c"], "b": ["e", "d"]}'
    # result = community_find(json.loads(raw))
    create_clients()

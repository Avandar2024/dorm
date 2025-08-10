import json
import os

import leidenalg as la
import pandas as pd
import requests
import tenacity
from igraph import Graph
from openai import OpenAI

from prompt import prompts_message, review_messages


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def content_review(data: dict) -> dict:
    """内容审核"""
    api_key = os.getenv('SILI_API_KEY')
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
    "model": "Pro/deepseek-ai/DeepSeek-V3",
    "enable_thinking": True,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "messages": [
        {
            "content": f"{review_messages(data)}",
            "role": "user"
        }
    ]
}
    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Content review failed with status code {response.status_code}: {response.text}")
    response_data = response.json()
    return response_data['choices'][0]['message']['content']


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def llm_process(data: dict) -> str:
    """使用LLM对数据进行分类"""
    api_key = os.getenv('DEEPSEEK_API_KEY')
    client = OpenAI(api_key=api_key, base_url='https://api.deepseek.com')
    reviewed_data = content_review(data)
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=prompts_message(reviewed_data),
        response_format={
            'type': 'json_object',
        },
    )
    json_output = response.choices[0].message.content
    if not json_output:
        raise ValueError('LLM did not return any content.')

    return json_output


def extract_json(raw: str) -> pd.DataFrame:
    """解析llm的输出"""
    group_info: dict = json.loads(raw)
    group_info = {str(k): [str(item) for item in v] for k, v in group_info.items()}
    # print(group_info)
    # 生成无向图
    g: Graph = Graph.ListDict(group_info, directed=False)
    # leidenalg社区发现
    partition = la.find_partition(g, la.ModularityVertexPartition)
    result = pd.DataFrame({'node': g.vs['name'], 'group': partition.membership})
    # print(result['group'])
    return result


def append_ai_col(dst: pd.DataFrame):
    """给DataFrame添加ai列"""
    ai_col: pd.Series = dst.iloc[:, -1]  # ai列是最后一列
    dst.drop(columns=dst.columns[-1], inplace=True)
    raw = llm_process(ai_col.to_dict())
    df = extract_json(raw)
    df.set_index('node', inplace=True)
    df.index = df.index.astype(dst.index.dtype)
    communities = df['group'].unique()
    for community in communities:
        new_col = f'ai_{community}'
        df[new_col] = df['group'] == community
        df[new_col].apply(lambda x: 1 if x else 0)
    df.drop(columns=['group'], inplace=True)
    dst = pd.concat([dst, df], axis=1)


if __name__ == '__main__':
    # test
    raw = '{"a": ["b", "c"], "b": ["e", "d"]}'
    result = extract_json(raw)

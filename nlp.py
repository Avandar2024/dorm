import json
import numpy as np


def confidence_level() -> float:
    """计算置信度"""
    return 0.95


def preprocess():
    """利用LLE进行数据降维"""
    pass


def group():
    """使用k-means聚类"""
    pass


def llm_process() -> str:
    """使用LLM对数据进行分类"""
    return ''


def extract_json(raw: str = llm_process()) -> np.ndarray:
    """解析llm的输出"""
    group_info:dict = json.loads(raw)
    g = nx.DiGraph(group_info)
    
    return np.ndarray(1)

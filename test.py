import os
from openai import OpenAI



def num_in_title(title: str) -> int:
    """从标题中提取数字"""
    num = ''.join(filter(str.isdigit, title))
    return int(num) if num else 0

if __name__ == '__main__':
  print(f'{num_in_title('1、如果第一节课程时间在九点及以后，您可以接受的舍友起床时间为:')}')
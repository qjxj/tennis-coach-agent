#  get_assistant_list 获取聊天助手和知识库信息
#  create_ask_delete  创建提问和删除会话获取rag查询结果
from typing import Annotated
from langchain_core.tools import tool
# 导入依赖
from ragflow_sdk import RAGFlow #链接rag服务的客户端

from api.monitor import monitor
from rawflow.rag_config import _load_ragflow_env

# 创建一个ragflow的客户端
api_key , base_url =_load_ragflow_env()
ragflow_client = RAGFlow(api_key=api_key,base_url=base_url)



# 1. 查询现在知识库中有哪些聊天助手和对应知识库的信息 （方便我们知道rag可以给我们提供哪些数据）
@tool
def get_assistant_list() -> str:
    """
    调用此工具可以查询 ragflow 服务器中有哪些助手，以及每个助手关联的知识库。
    向某个助手提问之前，必须先调用本工具拿到助手名称。
    """

    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    monitor.report_tool(tool_name="ragflow聊天助手列表查询工具：get_assistant_list")

    # 1. 创建ragflow客户端
    try:
        # 2. ragflow客户端查询所有的聊天助手 page: int = 1, page_size: int = 30
        chat_list = ragflow_client.list_chats()
        if not chat_list:
            return "没有任何可用助手"
        # 3. 查询聊天助手的知识库信息
        count_chat_info = "" #存储所有会话信息
        for chat in chat_list:
            dataset_names = []
            dataset_list = chat.datasets #当前聊天助手关联的知识库
            if dataset_list and isinstance(dataset_list,list):
                # 知识库的name
                for dataset in dataset_list:
                    # print(dataset)
                    dataset_names.append(dataset['name']) # 将一个助手的知识库的名字加入到列表中

            # 拼接格式: 助手名称 + 介绍 + 关联知识库列表
            count_chat_info += f"助手名称:{chat.name};功能介绍：{chat.description}; 关联的知识库：{'、'.join(dataset_names)} \n"
        return count_chat_info
    except Exception as e:
        return f"查询助手信息异常，无可用助手,异常信息:{str(e)}"

# 2. 对某个助手进行提问（创建会话 -》 提问 -》 删除会话）
@tool
def create_ask_delete(
    chat_name: Annotated[str, "助手名称，必须来自 get_assistant_list 的返回结果"],
    question: Annotated[str, "本次要向该助手提出的问题"],
)->str:
    """
    向指定助手创建一次性会话进行提问，提问结束后自动关闭会话，返回检索结果。

    调用前必须先调用 get_assistant_list 获取助手名称。
    """
    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    monitor.report_tool(tool_name="ragflow提问助手工具：create_ask_delete",args={"chat_name":chat_name,"question":question})
    # 1. 创建ragflow客户端
    # 2. 查询对应name的chat
    try:
        chats = ragflow_client.list_chats(name=chat_name)
        use_chat = chats[0] #选中我们要使用的助手
        # 3. chat上创建一个会话
        session = use_chat.create_session(name="temp_session_ask")
        # 4. 使用会话进行提问
        # 返回的提问结果是流式
        response = session.ask(question = question,stream=True)
        # 接收总结果
        result = ""
        # 流的每一部分的对象 part
        for part in response:
            # 数据存在对象中content上！！
            # print(part.content)
            result = part.content
        # 5. 关闭提问的会话
        # chat -> 关闭 -》  session
        use_chat.delete_sessions(ids=[session.id])
        # 6. 返回结果
        return result
    except Exception as e:
        return f"提问失败，错误原因：{str(e)}"

# if __name__ == '__main__':
#     # print(get_assistant_list())
#     # print(create_ask_delete("你的助手名称", "你的问题"))
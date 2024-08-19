import os
import pandas as pd
import dgl
import torch


# 加载数据
def load_data(dataset_name: str):
    file_path = os.path.join('./data', dataset_name, '{}.txt'.format(dataset_name))  # 文件路径
    if not os.path.exists(file_path):
        print('File not found')  # 文件不存在
        return None

    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    return df


# Twitter 数据处理
def data_processing_for_twitter(df: pd.DataFrame, snapshots=7):
    df.columns = ['userA', 'userB', 'timestamp', 'edge_type']
    df_list = []
    hetero_graph_list = []

    # 处理时间, 对时间戳分段
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    time_slot = (max_time - min_time + snapshots - 1) // snapshots  # 向上取整
    df['timestamp'] = df['timestamp'].apply(lambda x: (x - min_time) // time_slot)

    # 处理节点序号, 使之连续
    hash_map = {}  # 序号映射 map
    num = 0  # 初始序号
    for index, row in df.iterrows():
        if row['userA'] not in hash_map:
            hash_map[row['userA']] = num
            num += 1
        if row['userB'] not in hash_map:
            hash_map[row['userB']] = num
            num += 1
    df['userA'] = df['userA'].map(hash_map)
    df['userB'] = df['userB'].map(hash_map)

    # 定义节点和边类型
    node_types = ['user']
    edge_types = [('user', 'retweet', 'user'), ('user', 'mention', 'user'), ('user', 'reply', 'user')]
    edge_map = {'RT': 'retweet', 'MT': 'mention', 'RE': 'reply'}
    # 构造异质动态图
    for index in range(snapshots):
        # 对每个时间段构造异质图
        df_curr = df[df['timestamp'] == index]  # 取当前时间段的数据
        df_list.append(df_curr)
        df_retweet = df_curr[df_curr['edge_type'] == 'RT']  # retweet 类型的边
        df_mention = df_curr[df_curr['edge_type'] == 'MT']  # mention 类型的边
        df_reply = df_curr[df_curr['edge_type'] == 'RE']  # reply 类型的边
        # 定义每种类型的边
        data_dict = {
            ('user', 'retweet', 'user'): (torch.tensor(df_retweet['userA'].to_numpy()), torch.tensor(df_retweet['userB'].to_numpy())),
            ('user', 'mention', 'user'): (torch.tensor(df_mention['userA'].to_numpy()), torch.tensor(df_mention['userB'].to_numpy())),
            ('user', 'reply', 'user'): (torch.tensor(df_reply['userA'].to_numpy()), torch.tensor(df_reply['userB'].to_numpy())),
        }
        # 创建异构图
        hetero_graph = dgl.heterograph(data_dict)
        # 异构图预处理
        hetero_graph = dgl.to_simple(hetero_graph)  # 简化
        hetero_graph = dgl.to_bidirected(hetero_graph, copy_ndata=True)  # 双向化
        # 添加至列表
        hetero_graph_list.append(hetero_graph)
    print(hetero_graph_list)
    dgl.save_graphs(os.path.join('./data', 'Twitter', 'Twitter.bin'), hetero_graph_list)  # 保存
    print('Hetero graph list has been saved')
    return hetero_graph_list


# Math-Overflow 数据处理
def data_processing_for_math_overflow(df: pd.DataFrame, snapshots=11):
    df.columns = ['userA', 'userB', 'timestamp', 'edge_type']
    df_list = []
    hetero_graph_list = []

    # 处理时间, 对时间戳分段
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    time_slot = (max_time - min_time + snapshots - 1) // snapshots  # 向上取整
    df['timestamp'] = df['timestamp'].apply(lambda x: (x - min_time) // time_slot)

    # 处理节点序号, 使之连续
    hash_map = {}  # 序号映射 map
    num = 0  # 初始序号
    for index, row in df.iterrows():
        if row['userA'] not in hash_map:
            hash_map[row['userA']] = num
            num += 1
        if row['userB'] not in hash_map:
            hash_map[row['userB']] = num
            num += 1
    df['userA'] = df['userA'].map(hash_map)
    df['userB'] = df['userB'].map(hash_map)

    # 定义节点和边类型
    node_types = ['user']
    edge_types = [('user', 'answer_to_questions', 'user'), ('user', 'comment_to_answers', 'user'),
                  ('user', 'comment_to_questions', 'user')]
    edge_map = {'a2q': 'answer_to_questions', 'c2a': 'comment_to_answers', 'c2q': 'comment_to_questions'}
    # 构造异质动态图
    for index in range(snapshots):
        # 对每个时间段构造异质图
        df_curr = df[df['timestamp'] == index]  # 取当前时间段的数据
        df_list.append(df_curr)
        df_a2q = df_curr[df_curr['edge_type'] == 'a2q']  # answer_to_questions 类型的边
        df_c2a = df_curr[df_curr['edge_type'] == 'c2a']  # comment_to_answers 类型的边
        df_c2q = df_curr[df_curr['edge_type'] == 'c2q']  # comment_to_questions 类型的边
        # 定义每种类型的边
        data_dict = {
            ('user', 'answer_to_questions', 'user'): (torch.tensor(df_a2q['userA'].to_numpy()), torch.tensor(df_a2q['userB'].to_numpy())),
            ('user', 'comment_to_answers', 'user'): (torch.tensor(df_c2a['userA'].to_numpy()), torch.tensor(df_c2a['userB'].to_numpy())),
            ('user', 'comment_to_questions', 'user'): (torch.tensor(df_c2q['userA'].to_numpy()), torch.tensor(df_c2q['userB'].to_numpy())),
        }
        # 创建异构图
        hetero_graph = dgl.heterograph(data_dict)
        # 异构图预处理
        hetero_graph = dgl.to_simple(hetero_graph)  # 简化
        hetero_graph = dgl.to_bidirected(hetero_graph, copy_ndata=True)  # 双向化
        # 添加至列表
        hetero_graph_list.append(hetero_graph)
    print(hetero_graph_list)
    dgl.save_graphs(os.path.join('./data', 'MathOverflow', 'MathOverflow.bin'), hetero_graph_list)  # 保存
    print('Hetero graph list has been saved')
    return hetero_graph_list


# 获取 Twitter 数据
def get_twitter(snapshots=7):
    df = load_data('Twitter')
    hetero_graph_list = data_processing_for_twitter(df, snapshots)
    return hetero_graph_list


# 获取 Math-Overflow 数据
def get_math_overflow(snapshots=11):
    df = load_data('MathOverflow')
    hetero_graph_list = data_processing_for_math_overflow(df, snapshots)
    return hetero_graph_list

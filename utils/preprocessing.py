import os


# Math-Overflow 预处理
def preprocessing_for_math_overflow():
    file_paths = [os.path.join('./data', 'MathOverflow', 'a2q.txt'),
                  os.path.join('./data', 'MathOverflow', 'c2a.txt'),
                  os.path.join('./data', 'MathOverflow', 'c2q.txt')]  # 文件路径
    edge_types = ['a2q', 'c2a', 'c2q']  # 边的种类
    for file_path, edge_type in zip(file_paths, edge_types):
        # 读取文件的所有行
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 在每一行的末尾添加指定的单词
        modified_lines = [line.strip() + ' ' + edge_type for line in lines]
        # 加入换行符，但最后一行不加
        modified_content = '\n'.join(modified_lines)
        # 将修改后的行写回到原文件中
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_content)
    # 合并三个文件
    output_file = os.path.join('./data', 'MathOverflow', 'MathOverflow.txt')
    # 打开并读取第一个文件
    with open(file_paths[0], 'r', encoding='utf-8') as f1:
        content1 = f1.read()

    # 打开并读取第二个文件
    with open(file_paths[1], 'r', encoding='utf-8') as f2:
        content2 = f2.read()

    # 打开并读取第三个文件
    with open(file_paths[2], 'r', encoding='utf-8') as f3:
        content3 = f3.read()

    # 将所有内容写入新的目标文件
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(content1)
        out_file.write('\n')  # 添加换行符以分隔文件内容
        out_file.write(content2)
        out_file.write('\n')  # 添加换行符以分隔文件内容
        out_file.write(content3)

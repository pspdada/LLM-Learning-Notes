import os
import re
from pathlib import Path

## 由于 Notion 导出的文件名和文件夹名中包含了随机生成的 ID，因此需要将这些 ID 去掉。

# 获取当前工作目录
current_directory = Path.cwd()


def remove_random_ids():
    """
    Removes the random IDs from filenames and directory names in the current directory.
    """
    directory = current_directory / 'LLM 学习记录'
    print(directory)

    # 遍历所有的 .md 文件
    for file_path in directory.glob('*.md'):
        new_name = re.sub(r'(.*) ([a-zA-Z0-9]+)\.md$', r'\1.md', file_path.name)
        if new_name != file_path.name:
            new_file_path = file_path.with_name(new_name)
            file_path.rename(new_file_path)

    # 遍历所有的子目录
    for folder_path in directory.iterdir():
        if folder_path.is_dir():
            new_name = re.sub(r'(.*) ([a-zA-Z0-9]+)$', r'\1', folder_path.name)
            if new_name != folder_path.name:
                new_folder_path = folder_path.with_name(new_name)
                folder_path.rename(new_folder_path)


def update_md_links():
    """
    Updates links within .md files to reflect the removal of random IDs.
    """

    # 遍历所有的 .md 文件
    for file_path in current_directory.glob('*.md'):
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()

            # 匹配并替换文件内部的链接
            content, num_subs = re.subn(r'\[(.*?)\]\(([^:]+)%20([a-zA-Z0-9]+)\/([^:]+)%20([a-zA-Z0-9]+)\.md\)',
                                        lambda m: f"[{m.group(1)}]({m.group(2)}/{m.group(4)}.md)", content)

            # 再次匹配并替换文件内部的链接，这次不包含 %20
            content, num_subs = re.subn(r'\[(.*?)\]\(([^:]+)%20([a-zA-Z0-9]+)\.md\)',
                                        lambda m: f"[{m.group(1)}]({m.group(2)}.md)", content)

            # 更新文件内容
            file.seek(0)
            file.write(content)
            file.truncate()


if __name__ == "__main__":
    # 移除随机编号
    remove_random_ids()
    # 更新 .md 文件中的链接
    update_md_links()
    print("Processing complete.")

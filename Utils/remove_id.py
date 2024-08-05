import os
import re
from pathlib import Path
import shutil

# 由于 Notion 保存的文件名中包含了随机 ID，这个脚本用于移除这些 ID，并且修正 .md 文件中的引用

# 获取当前工作目录
current_directory = Path.cwd()

# 正则表达式，用于匹配文件和文件夹名中的随机ID
id_pattern = re.compile(r'(%20|\s)\w{32}')


# 处理资源文件夹和 .md 文件的名称
def remove_ids_from_names():
    # 遍历 LLM 学习记录 文件夹
    llm_folder = current_directory / "LLM 学习记录"
    for item in llm_folder.iterdir():
        # 检查是否是资源文件夹或 .md 文件
        if item.is_dir() or item.suffix == ".md":
            # 去除名称中的ID
            new_name = re.sub(id_pattern, '', item.name)
            new_path = item.with_name(new_name)

            # 检查新路径是否已经存在
            if new_path.exists():
                if item.is_dir():
                    # 如果是文件夹，将内容复制（覆盖）到无ID的文件夹
                    for sub_item in item.iterdir():
                        dest = new_path / sub_item.name
                        if sub_item.is_file():
                            shutil.copy2(sub_item, dest)
                        elif sub_item.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(sub_item, dest)
                    # 删除有ID的文件夹
                    shutil.rmtree(item)
                    print(f"Copied and removed: {item.name} -> {new_name}")
                else:
                    # 如果是文件且目标文件已存在，暂时不处理
                    print(f"Skipping: {item.name} -> {new_name} (file already exists)")
            else:
                # 重命名文件或文件夹
                item.rename(new_path)
                print(f"Renamed: {item.name} -> {new_name}")


# 修正 .md 文件中的引用
def fix_md_references():
    llm_folder = current_directory / "LLM 学习记录"
    for md_file in llm_folder.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        new_content = re.sub(id_pattern, '', content)
        if new_content != content:
            md_file.write_text(new_content, encoding="utf-8")
            print(f"Updated references in: {md_file.name}")


if __name__ == "__main__":
    remove_ids_from_names()
    fix_md_references()
    print("Processing complete.")

"""
環境ファイルをリセットして配置しなおすためのスクリプト
"""

import os
import shutil
import parse

TARGET_PATH = "."
ENV_FILES_FOLDER = "new_npz_env_files"
EXTENSION = ".npz"
ENV_FILE_PARSER = parse.compile("{}agents_{}size_{}density_{}wall_id{}" + EXTENSION)
FIRST_FOLDER_FORMAT = "{}size_{}density_{}wall"

ENV_FILES_PATH = os.path.join(TARGET_PATH, ENV_FILES_FOLDER)


# 環境ファイルのリセット
def reset_env_files():
    print("Resetting env files...")

    # 格納するフォルダの作成
    os.makedirs(ENV_FILES_PATH, exist_ok=True)

    all_env_files_path = [
        (os.path.join(pathname, filename), filename)
        for pathname, _, filenames in os.walk(TARGET_PATH)
        for filename in filenames
        if filename.endswith(EXTENSION)
    ]

    moved_env_files = []

    for env_file_path, env_file_name in all_env_files_path:
        # 重複したファイルの削除
        if env_file_name in moved_env_files:
            os.remove(env_file_path)
            continue

        # ファイル名のパース
        parsed = ENV_FILE_PARSER.parse(env_file_name)
        parsed_list = list(parsed)
        if parsed is None or len(parsed_list) != 5:
            raise ValueError(f"Invalid filename: {env_file_name}")

        # ファイルの移動先のパスを作成
        first_folder = FIRST_FOLDER_FORMAT.format(*parsed_list[1:4])  # like "20size_0.3density_1wall"
        second_folder = str(parsed_list[0])  # like "4"
        dst_folder_path = os.path.join(ENV_FILES_PATH, first_folder, second_folder)
        moved_env_file_path = os.path.join(dst_folder_path, env_file_name)

        # フォルダの作成
        os.makedirs(dst_folder_path, exist_ok=True)

        # ファイルの移動
        shutil.move(env_file_path, moved_env_file_path)
        moved_env_files.append(env_file_name)

    removed_folders = []
    # 空のフォルダの削除
    for pathname, _, _ in os.walk(TARGET_PATH, topdown=False):
        if not os.listdir(pathname):
            os.rmdir(pathname)
            removed_folders.append(pathname)

    print("Resetting env files is done.")
    print(f"Moved env files: {len(moved_env_files)}")
    print(f"Removed folders: {len(removed_folders)}")


def main():
    reset_env_files()


if __name__ == "__main__":
    main()

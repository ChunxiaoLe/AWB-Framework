import os
import numpy as np
from pathlib import Path
import glob


def read_and_merge_txt_files(data_fold_path):
    """
    读取data_fold中所有相机文件夹的txt文件并合并
    """
    all_files = []
    camera_folders = []

    # 遍历data_fold目录下的所有子文件夹
    for camera_folder in os.listdir(data_fold_path):
        camera_path = os.path.join(data_fold_path, camera_folder)

        # 检查是否为文件夹
        if os.path.isdir(camera_path):
            camera_folders.append(camera_folder)

            # 查找该相机文件夹下的txt文件
            txt_files = glob.glob(os.path.join(camera_path, "*.txt"))

            for txt_file in txt_files:
                print(f"读取文件: {txt_file}")
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:  # 忽略空行
                                # 移除文件扩展名，获取基础文件名
                                base_name = os.path.splitext(line)[0]
                                all_files.append({
                                    'original_name': line,
                                    'base_name': base_name,
                                    'camera': camera_folder,
                                    'source_txt': txt_file
                                })
                except Exception as e:
                    print(f"读取文件 {txt_file} 时出错: {e}")

    print(f"找到 {len(camera_folders)} 个相机文件夹: {camera_folders}")
    print(f"总共合并了 {len(all_files)} 个文件记录")
    return all_files, camera_folders


def check_npy_files(file_info, data_fold_path):
    """
    检查对应的.npy文件是否存在并能正确读取
    """
    base_name = file_info['base_name']
    camera = file_info['camera']
    camera_path = '/dataset/NUS_full'

    # 需要检查的三个文件类型
    file_types = ['', '_camera', '_gt', '_mask']  # 根据图片调整了文件类型

    results = {
        'missing_files': [],
        'unreadable_files': [],
        'success_files': []
    }

    for file_type in file_types:
        npy_filename = f"{base_name}{file_type}.npy"
        npy_path = os.path.join(camera_path, npy_filename)

        # 检查文件是否存在
        if not os.path.exists(npy_path):
            results['missing_files'].append(npy_filename)
        else:
            # 尝试读取文件
            try:
                data = np.load(npy_path)
                results['success_files'].append(npy_filename)
                # 可以添加更多的验证，比如检查数据形状、类型等
                # print(f"成功读取 {npy_filename}, 形状: {data.shape}")
            except Exception as e:
                results['unreadable_files'].append({
                    'filename': npy_filename,
                    'error': str(e)
                })

    return results


def save_problem_files_to_txt(total_missing, total_unreadable, output_file="problem_files.txt"):
    """
    将问题文件保存到txt文件中
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("数据集文件检查结果 - 问题文件报告\n")
            f.write("=" * 60 + "\n\n")

            # 写入缺失文件
            f.write(f"缺失的文件 (总数: {len(total_missing)})\n")
            f.write("-" * 60 + "\n")
            if total_missing:
                for item in total_missing:
                    f.write(f"相机: {item['camera']}\n")
                    f.write(f"原文件: {item['original_name']}\n")
                    f.write(f"缺失文件: {item['missing_file']}\n")
                    f.write(f"完整路径: {os.path.join(item['camera'], item['missing_file'])}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("无缺失文件\n")

            f.write("\n" + "=" * 60 + "\n\n")

            # 写入无法读取的文件
            f.write(f"无法读取的文件 (总数: {len(total_unreadable)})\n")
            f.write("-" * 60 + "\n")
            if total_unreadable:
                for item in total_unreadable:
                    f.write(f"相机: {item['camera']}\n")
                    f.write(f"原文件: {item['original_name']}\n")
                    f.write(f"问题文件: {item['unreadable_file']}\n")
                    f.write(f"完整路径: {os.path.join(item['camera'], item['unreadable_file'])}\n")
                    f.write(f"错误信息: {item['error']}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("无无法读取的文件\n")

            # 写入简要统计
            f.write("\n" + "=" * 60 + "\n")
            f.write("统计摘要\n")
            f.write("-" * 60 + "\n")
            f.write(f"缺失文件总数: {len(total_missing)}\n")
            f.write(f"无法读取文件总数: {len(total_unreadable)}\n")
            f.write(f"问题文件总数: {len(total_missing) + len(total_unreadable)}\n")

            # 按文件类型统计缺失情况
            missing_by_type = {}
            for item in total_missing:
                file_type = item['missing_file'].split('_')[-1].replace('.npy', '')
                if '_' not in item['missing_file']:
                    file_type = 'base'
                missing_by_type[file_type] = missing_by_type.get(file_type, 0) + 1

            f.write(f"\n按文件类型统计缺失情况:\n")
            for file_type, count in missing_by_type.items():
                f.write(f"  {file_type}: {count} 个\n")

        print(f"问题文件报告已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"保存问题文件报告时出错: {e}")
        return False


def main():
    # 数据集路径 - 根据你的实际路径修改
    data_fold_path = "/dataset/NUS_full/data_fold"

    # 如果路径不存在，尝试相对路径
    if not os.path.exists(data_fold_path):
        data_fold_path = "./data_fold"
        if not os.path.exists(data_fold_path):
            print(f"错误: 找不到数据集路径。请修改脚本中的 data_fold_path 变量")
            return

    print(f"开始检查数据集: {data_fold_path}")
    print("=" * 60)

    # 第一步：读取并合并所有txt文件
    all_files, camera_folders = read_and_merge_txt_files(data_fold_path)

    if not all_files:
        print("没有找到任何文件记录，请检查路径和txt文件")
        return

    print("\n" + "=" * 60)
    print("开始检查.npy文件...")

    # 统计信息
    total_missing = []
    total_unreadable = []
    total_success = 0

    # 第二步：检查每个文件对应的.npy文件
    for i, file_info in enumerate(all_files):
        if i % 100 == 0:  # 每100个文件显示一次进度
            print(f"检查进度: {i + 1}/{len(all_files)}")

        results = check_npy_files(file_info, data_fold_path)

        # 收集统计信息
        if results['missing_files']:
            for missing_file in results['missing_files']:
                total_missing.append({
                    'camera': file_info['camera'],
                    'original_name': file_info['original_name'],
                    'missing_file': missing_file
                })

        if results['unreadable_files']:
            for unreadable_info in results['unreadable_files']:
                total_unreadable.append({
                    'camera': file_info['camera'],
                    'original_name': file_info['original_name'],
                    'unreadable_file': unreadable_info['filename'],
                    'error': unreadable_info['error']
                })

        total_success += len(results['success_files'])

    # 第三步：输出结果
    print("\n" + "=" * 60)
    print("检查结果汇总:")
    print("=" * 60)

    print(f"总文件数: {len(all_files)}")
    print(f"成功读取的.npy文件数: {total_success}")
    print(f"缺失的文件数: {len(total_missing)}")
    print(f"无法读取的文件数: {len(total_unreadable)}")

    # 输出缺失的文件
    if total_missing:
        print(f"\n缺失的文件 ({len(total_missing)} 个):")
        print("-" * 40)
        for item in total_missing[:20]:  # 只显示前20个，避免输出过长
            print(f"相机: {item['camera']}, 原文件: {item['original_name']}, 缺失: {item['missing_file']}")
        if len(total_missing) > 20:
            print(f"... 还有 {len(total_missing) - 20} 个缺失文件")

    # 输出无法读取的文件
    if total_unreadable:
        print(f"\n无法读取的文件 ({len(total_unreadable)} 个):")
        print("-" * 40)
        for item in total_unreadable[:10]:  # 只显示前10个
            print(f"相机: {item['camera']}, 原文件: {item['original_name']}")
            print(f"无法读取: {item['unreadable_file']}")
            print(f"错误: {item['error']}")
            print()
        if len(total_unreadable) > 10:
            print(f"... 还有 {len(total_unreadable) - 10} 个无法读取的文件")

    # 按相机分类统计
    print(f"\n按相机分类的统计:")
    print("-" * 40)
    for camera in camera_folders:
        camera_missing = [item for item in total_missing if item['camera'] == camera]
        camera_unreadable = [item for item in total_unreadable if item['camera'] == camera]
        camera_total = len([item for item in all_files if item['camera'] == camera])

        print(f"{camera}: 总数={camera_total}, 缺失={len(camera_missing)}, 无法读取={len(camera_unreadable)}")

    # 保存问题文件到txt文件
    print("\n" + "=" * 60)
    print("保存问题文件报告...")

    # 生成输出文件名（包含时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"problem_files_{timestamp}.txt"

    success = save_problem_files_to_txt(total_missing, total_unreadable, output_filename)

    if success:
        print(f"✓ 问题文件报告已成功保存到: {output_filename}")
        print(f"  - 缺失文件: {len(total_missing)} 个")
        print(f"  - 无法读取文件: {len(total_unreadable)} 个")
        print(f"  - 总问题文件: {len(total_missing) + len(total_unreadable)} 个")
    else:
        print("✗ 保存问题文件报告失败")

    print("\n检查完成!")


if __name__ == "__main__":
    main()
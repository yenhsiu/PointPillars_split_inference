"""
TFLite 模型延迟测量脚本 - 简化版
专门用于 Raspberry Pi 4 收集推理延迟数据

功能:
- 测量推理延迟 (latency)
- 保存原始数据到 CSV
- 简单统计信息
"""

import tensorflow as tf
import numpy as np
import time
import csv
import os
import argparse
from datetime import datetime


def measure_latency(model_path, num_iterations=100, num_warmup=10, num_threads=4):
    """
    测量 TFLite 模型推理延迟
    
    Args:
        model_path: TFLite 模型路径
        num_iterations: 测试次数
        num_warmup: 预热次数
        num_threads: 使用的线程数
    
    Returns:
        latencies: 延迟列表 (ms)
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=num_threads
    )
    interpreter.allocate_tensors()
    
    # 获取输入输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"输入形状: {input_shape}")
    print(f"输出数量: {len(output_details)}")
    print(f"线程数: {num_threads}")
    
    # 生成测试数据
    test_input = np.random.randn(*input_shape).astype(input_dtype)
    
    # 预热
    print(f"\n预热 {num_warmup} 次...")
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
    print("预热完成")
    
    # 测量延迟
    print(f"\n开始测量 ({num_iterations} 次)...")
    latencies = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 20 == 0:
            print(f"  完成 {i+1}/{num_iterations}")
    
    print("测量完成\n")
    
    return latencies


def print_statistics(latencies):
    """打印统计信息"""
    latencies = np.array(latencies)
    
    print("="*50)
    print("延迟统计")
    print("="*50)
    print(f"样本数: {len(latencies)}")
    print(f"平均值: {latencies.mean():.2f} ms")
    print(f"中位数: {np.median(latencies):.2f} ms")
    print(f"标准差: {latencies.std():.2f} ms")
    print(f"最小值: {latencies.min():.2f} ms")
    print(f"最大值: {latencies.max():.2f} ms")
    print(f"P50: {np.percentile(latencies, 50):.2f} ms")
    print(f"P90: {np.percentile(latencies, 90):.2f} ms")
    print(f"P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99: {np.percentile(latencies, 99):.2f} ms")
    print(f"吞吐量: {1000/latencies.mean():.2f} FPS")
    print("="*50)


def save_results(latencies, model_path, save_dir='latency_results'):
    """保存延迟数据到 CSV"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_file = os.path.join(save_dir, f"{model_name}_latency_{timestamp}.csv")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'latency_ms'])
        for i, lat in enumerate(latencies, 1):
            writer.writerow([i, f"{lat:.4f}"])
    
    print(f"\n结果已保存到: {csv_file}")
    return csv_file


def main():
    parser = argparse.ArgumentParser(description='TFLite 延迟测量 - 简化版')
    parser.add_argument('--model', required=True,
                        help='TFLite 模型路径')
    parser.add_argument('--iterations', type=int, default=100,
                        help='测试次数 (默认: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数 (默认: 10)')
    parser.add_argument('--threads', type=int, default=4,
                        help='线程数 (默认: 4)')
    parser.add_argument('--save_dir', default='latency_results',
                        help='保存目录 (默认: latency_results)')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("TFLite 延迟测量工具")
    print("="*50 + "\n")
    
    try:
        # 测量延迟
        latencies = measure_latency(
            model_path=args.model,
            num_iterations=args.iterations,
            num_warmup=args.warmup,
            num_threads=args.threads
        )
        
        # 打印统计
        print_statistics(latencies)
        
        # 保存结果
        csv_file = save_results(latencies, args.model, args.save_dir)
        
        print("\n✅ 完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

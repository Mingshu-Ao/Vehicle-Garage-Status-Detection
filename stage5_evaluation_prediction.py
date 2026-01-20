# stage5_evaluation_prediction.py
# 第五阶段：模型优化、多维评估与最终预测 (路径修正版)

import numpy as np
import pandas as pd
import os
import time
import pickle
import warnings
import sys

# 机器学习模型
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ================= 1. 简化的流水线重建 (Pipeline Reconstruction) =================
class EvaluationPipeline:
    """重建从 原始数据 -> SVD -> 特征筛选 的完整流水线"""
    
    def __init__(self, train_file):
        self.train_file = train_file
        self.mean = None
        self.std = None
        self.V_k = None # SVD 投影矩阵
        self.selected_indices = None # 特征筛选索引
        self.scaler = None # 标准化器
        self.models = {}
        self.X_train_final = None
        self.y_train = None
        self.valid_columns = None
        
    def fit_pipeline(self):
        """重新拟合预处理参数 (为了处理测试集)"""
        print(">>> [Pipeline] 正在重建数据预处理流水线...")
        
        # 1. 加载原始训练数据
        if not os.path.exists(self.train_file):
            print(f"错误: 找不到文件 {self.train_file}")
            return False
            
        print(f"   -> 读取文件: {self.train_file}")
        try:
            df = pd.read_csv(self.train_file)
        except Exception as e:
            print(f"   -> 读取失败: {e}")
            return False
        
        # 2. 清洗逻辑 (与阶段2一致)
        drop_cols = ['time', 'group_name', 'light_is_daytime']
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # 强力清洗
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_num = df[numeric_cols].fillna(df[numeric_cols].mean()).fillna(0)
        
        # 剔除高方差和常量列
        variances = df_num.var()
        bad_cols = variances[(variances > 1e9) | (variances == 0)].index
        if len(bad_cols) > 0:
            df_num = df_num.drop(columns=bad_cols)
            
        self.valid_columns = df_num.columns.drop('labelArea') if 'labelArea' in df_num.columns else df_num.columns
        
        X = df_num.drop(columns=['labelArea']).values
        y = df_num['labelArea'].values
        
        # 3. 拟合 SVD (简化版)
        print("   -> 拟合 SVD 参数...")
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        X_std = (X - self.mean) / self.std
        
        # 快速 SVD
        # 使用 numpy.linalg.svd 计算相关性矩阵的特征向量
        cov_matrix = (X_std.T @ X_std) / (X.shape[0]-1)
        U, S, Vt = np.linalg.svd(cov_matrix)
        
        # 假设阶段2选了 k=80 (根据你的日志)
        k = min(80, X.shape[1])
        self.V_k = Vt.T[:, :k] # (n_features, k)
        
        X_svd = X_std @ self.V_k
        
        X_final = X_svd
        
        # 5. 拟合标准化器
        self.scaler = StandardScaler()
        self.X_train_final = self.scaler.fit_transform(X_final)
        self.y_train = y
        
        print(">>> [Pipeline] 流水线重建完成。")
        return True

    def train_models(self):
        """快速重新训练模型"""
        print(">>> [Models] 正在重新训练模型...")
        
        # 1. 随机森林 (RF)
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(self.X_train_final, self.y_train)
        self.models['RandomForest'] = rf
        
        # 2. 神经网络 (MLP)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        mlp.fit(self.X_train_final, self.y_train)
        self.models['NeuralNet'] = mlp
        
        # 3. 逻辑回归 (模拟 LU-LR)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(solver='newton-cg', max_iter=20)
        lr.fit(self.X_train_final, self.y_train)
        self.models['LU-Logistic'] = lr
        
        print(">>> [Models] 模型准备就绪。")

    def process_test_data(self, raw_df):
        """将原始测试数据通过流水线转换为模型输入"""
        # 1. 对齐列 (缺失的列补0，多余的列忽略)
        #创建一个全0的DataFrame作为模板
        df_clean = pd.DataFrame(0, index=raw_df.index, columns=self.valid_columns)
        # 更新存在的列
        common_cols = raw_df.columns.intersection(self.valid_columns)
        df_clean[common_cols] = raw_df[common_cols]
        
        X = df_clean.values
        
        # 2. 缺失值处理
        X = np.nan_to_num(X)
        
        # 3. 标准化 + SVD 投影
        X_std = (X - self.mean) / self.std
        X_svd = X_std @ self.V_k
        
        # 4. 最终标准化
        X_final = self.scaler.transform(X_svd)
        return X_final

# ================= 2. 四维评估指标计算器 =================
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(model, X_test, y_test, model_name):
        results = {}
        
        # --- 指标 1: 计算效率 (推理耗时) ---
        start_t = time.time()
        # 跑多次取平均
        for _ in range(5): 
            _ = model.predict(X_test[:100])
        avg_infer_time_ms = (time.time() - start_t) / 5 * 1000 
        results['Inference_Time_1k_ms'] = avg_infer_time_ms
        
        # --- 指标 2: 模型大小 ---
        dump = pickle.dumps(model)
        size_mb = len(dump) / 1024 / 1024
        results['Model_Size_MB'] = size_mb
        
        # --- 指标 3 & 4: 实时性与有效准确率 ---
        y_pred = model.predict(X_test)
        
        # 寻找状态切换点
        events_true = []
        for i in range(1, len(y_test)):
            if y_test[i-1] == 0 and y_test[i] != 0:
                events_true.append({'idx': i, 'class': y_test[i]})
        
        detected_count = 0
        total_latency = 0
        valid_events = 0
        
        for event in events_true:
            # 窗口：前 20s (-1000样本) 到 后 40s (+2000样本)
            start_search = max(0, event['idx'] - 1000)
            end_search = min(len(y_pred), event['idx'] + 2000)
            
            window_pred = y_pred[start_search:end_search]
            
            try:
                # 找到第一个预测为该类别的索引
                detect_offset = np.where(window_pred == event['class'])[0][0]
                detect_idx = start_search + detect_offset
                
                # 计算时延 (秒, 假设50Hz)
                latency = (detect_idx - event['idx']) * 0.02
                total_latency += abs(latency)
                detected_count += 1
                valid_events += 1
            except IndexError:
                pass
        
        base_acc = accuracy_score(y_test, y_pred)
        event_recall = detected_count / len(events_true) if len(events_true) > 0 else 0
        
        results['Effective_Accuracy'] = (base_acc * 0.6 + event_recall * 0.4)
        results['Avg_Latency_s'] = total_latency / valid_events if valid_events > 0 else 10.0
        
        return results

# ================= 3. 主程序 =================
def run_stage_5():
    print("="*60)
    print("第五阶段：模型优化、评估与最终预测")
    print("="*60)
    
    # ================= 路径配置 (已修正为绝对路径) =================
    base_dir = "D:/bupt/code/python/数值计算期末作业数据"
    train_path = os.path.join(base_dir, "train_data.csv")
    test_dir = os.path.join(base_dir, "test")
    output_dir = "prediction_results"  # 结果保存在当前脚本目录下
    # ==============================================================

    # 1. 初始化并运行流水线
    pipeline = EvaluationPipeline(train_file=train_path)
    if not pipeline.fit_pipeline():
        print("错误: 流水线初始化失败，请检查路径。")
        return
    
    pipeline.train_models()
    
    # 2. 评估模型
    print("\n" + "="*40)
    print("1. 多维指标评估")
    print("="*40)
    
    # 切分验证集
    X_train, X_eval, y_train, y_eval = train_test_split(
        pipeline.X_train_final, pipeline.y_train, test_size=0.2, shuffle=False
    )
    
    eval_report = {}
    for name, model in pipeline.models.items():
        print(f"评估模型: {name}...")
        metrics = ModelEvaluator.calculate_metrics(model, X_eval, y_eval, name)
        eval_report[name] = metrics
        
    # 打印评估表
    report_df = pd.DataFrame(eval_report).T
    print("\n评估结果汇总:")
    print(report_df)
    report_df.to_csv('final_model_evaluation.csv')
    
    # 选择最佳模型
    best_score = -float('inf')
    best_model_name = None
    
    # 简单的打分公式: 准确率权重高，时延越小越好
    for name, metrics in eval_report.items():
        score = metrics['Effective_Accuracy'] * 100 - metrics['Avg_Latency_s'] * 1
        if score > best_score:
            best_score = score
            best_model_name = name
            
    print(f"\n🏆 综合最佳模型: {best_model_name}")
    final_model = pipeline.models[best_model_name]
    
    # 3. 模型优化 (以随机森林剪枝为例)
    if best_model_name == 'RandomForest':
        print("\n" + "="*40)
        print("2. 模型优化 (剪枝)")
        print("="*40)
        print(f"原始大小: {eval_report['RandomForest']['Model_Size_MB']:.2f} MB")
        
        optimized_rf = RandomForestClassifier(n_estimators=20, max_depth=8, min_samples_leaf=5)
        optimized_rf.fit(X_train, y_train)
        
        opt_metrics = ModelEvaluator.calculate_metrics(optimized_rf, X_eval, y_eval, "RF_Optimized")
        print(f"优化后大小: {opt_metrics['Model_Size_MB']:.2f} MB")
        print(f"优化后准确率: {opt_metrics['Effective_Accuracy']:.4f}")
        
        if opt_metrics['Effective_Accuracy'] > 0.85:
            print("-> 采纳优化后的模型")
            final_model = optimized_rf
            
    # 4. 验证集最终预测
    print("\n" + "="*40)
    print("3. 执行最终预测 (test 文件夹)")
    print("="*40)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
        print(f"找到 {len(files)} 个测试文件，开始预测...")
        
        count = 0
        for f in files:
            file_path = os.path.join(test_dir, f)
            try:
                # 读取原始文件
                raw_df = pd.read_csv(file_path)
                
                # 预处理
                X_test = pipeline.process_test_data(raw_df)
                
                # 预测
                y_pred = final_model.predict(X_test)
                
                # 追加标签列
                result_df = raw_df.copy()
                result_df['labelArea'] = y_pred
                
                # 保存
                save_path = os.path.join(output_dir, f"pred_{f}")
                result_df.to_csv(save_path, index=False)
                count += 1
                if count % 5 == 0:
                    print(f"  已处理 {count}/{len(files)} 个文件...")
                
            except Exception as e:
                print(f"  处理文件 {f} 失败: {e}")
                
        print(f"\n成功！所有 {count} 个预测结果已保存在 '{output_dir}' 文件夹中。")
    else:
        print(f"警告: 未找到测试文件夹 {test_dir}")

    print("\n" + "="*60)
    print("🎉 全流程任务圆满完成！")
    print("="*60)

if __name__ == "__main__":
    run_stage_5()
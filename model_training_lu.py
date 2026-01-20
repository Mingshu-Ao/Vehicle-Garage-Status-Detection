# model_training_lu.py
# 第四阶段：模型设计与训练 (带列选主元的LU分解 + 高级算法对比)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import json

# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 高级算法对比
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.neural_network import MLPClassifier     # 简易神经网络(深度学习)

# 屏蔽警告
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
except:
    pass

# ================= 1. 数值计算核心：带选主元的 LU 分解 (PA=LU) =================
class NumericalSolver:
    """
    手动实现数值线性代数算法
    包含：带列选主元的 LU 分解 (Partial Pivoting)
    """
    
    @staticmethod
    def lu_decomposition_pivoting(A):
        """
        实现 PA = LU 分解 (列选主元，保证数值稳定性)
        输入: A (n x n 矩阵)
        输出: P (置换向量), L, U
        """
        n = A.shape[0]
        U = A.copy().astype(float)
        L = np.eye(n).astype(float)
        P = np.arange(n) # 置换记录向量
        
        for k in range(n - 1):
            # 1. 选主元 (Pivoting): 找到第 k 列中绝对值最大的行
            pivot_row = k + np.argmax(np.abs(U[k:, k]))
            
            # 如果主元太小，可能奇异，但继续计算
            if abs(U[pivot_row, k]) < 1e-12:
                continue
                
            # 2. 交换行 (在 U, L, P 中都要交换)
            if pivot_row != k:
                # 交换 U 的行
                U[[k, pivot_row], :] = U[[pivot_row, k], :]
                # 交换 L 的行 (仅交换左下角已计算部分)
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
                # 记录置换
                P[[k, pivot_row]] = P[[pivot_row, k]]
            
            # 3.由于 PA = LU，消元计算
            for i in range(k + 1, n):
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] = U[i, k:] - factor * U[k, k:]
                
        return P, L, U
    
    @staticmethod
    def solve_lu_pivoting(P, L, U, b):
        """
        求解 Ax = b -> PAx = Pb -> LUx = Pb
        1. b_new = Pb (应用置换)
        2. Ly = b_new (前代法)
        3. Ux = x (回代法)
        """
        n = len(b)
        x = np.zeros(n)
        y = np.zeros(n)
        
        # 1. 应用置换 P 到 b
        b_new = b[P]
        
        # 2. 前代法解 Ly = b_new
        for i in range(n):
            # y[i] = b_new[i] - sum(L[i,j]*y[j])
            y[i] = b_new[i] - np.dot(L[i, :i], y[:i])
            
        # 3. 回代法解 Ux = y
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < 1e-12:
                # 处理奇异矩阵情况，给予极小值防止除零
                U[i, i] = 1e-12 
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
            
        return x

# ================= 2. 手写模型：基于 LU 的逻辑回归 =================
class ManualLogisticRegression:
    """
    基于牛顿法 (Newton-Raphson) 的逻辑回归
    核心：使用手动实现的 LU 分解求解 Hessian * delta = Gradient
    """
    def __init__(self, max_iter=20, tol=1e-4, lambda_reg=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.weights = None
        self.losses = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        m, n = X.shape
        # 添加偏置项
        X_b = np.c_[np.ones((m, 1)), X]
        n_params = n + 1
        
        # 初始化权重
        self.weights = np.zeros(n_params)
        
        print(f"   [LU-LR] 开始训练 (Newton法, 正则化={self.lambda_reg})...")
        
        for i in range(self.max_iter):
            # 1. 前向计算
            z = X_b @ self.weights
            h = self.sigmoid(z)
            
            # 2. 计算梯度 (Gradient)
            # Grad = X.T * (h - y) + lambda * w
            gradient = X_b.T @ (h - y) + self.lambda_reg * self.weights
            
            # 3. 计算海森矩阵 (Hessian)
            # H = X.T * diag(h(1-h)) * X + lambda * I
            # 向量化加速计算
            W_vec = h * (1 - h)
            # 等价于 X.T @ W @ X
            H = X_b.T @ (W_vec[:, None] * X_b)
            # 添加正则化 (岭回归) 保证 H 正定可逆
            H += self.lambda_reg * np.eye(n_params)
            
            # 4. 关键步骤：使用 LU 分解求解线性方程组 H * delta = gradient
            try:
                # 调用手写的数值算法
                P, L, U = NumericalSolver.lu_decomposition_pivoting(H)
                delta = NumericalSolver.solve_lu_pivoting(P, L, U, gradient)
            except Exception as e:
                print(f"     警告: 迭代 {i} 出现数值错误 ({e})，停止迭代")
                break
            
            # 5. 更新权重
            self.weights -= delta
            
            # 6. 计算 Loss (交叉熵)
            epsilon = 1e-15
            loss = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
            self.losses.append(loss)
            
            grad_norm = np.linalg.norm(delta)
            if (i+1) % 5 == 0:
                print(f"     Iter {i+1}/{self.max_iter}: Loss={loss:.4f}, Update={grad_norm:.4f}")
            
            if grad_norm < self.tol:
                print("     -> 收敛！")
                break
                
        return self

    def predict_proba(self, X):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return self.sigmoid(X_b @ self.weights)

# ================= 3. 运行主流程 =================
def run_stage_4():
    print("="*60)
    print("第四阶段：模型设计与训练")
    print("目标：基于LU分解构建模型，并与随机森林/神经网络对比")
    print("="*60)
    
    # 1. 加载数据
    data_path = 'feature_selected_train.csv'
    if not os.path.exists(data_path):
        print(f"错误: 找不到 {data_path}，请先运行阶段3")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['labelArea']).values
    y = df['labelArea'].values
    
    print(f"加载数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y.astype(int))}")
    
    # 2. 切分数据集 (8:2)
    print("\n1. 数据集切分 (80% 训练, 20% 测试)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化 (对线性模型很重要)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    results = {}
    
    # ================= 模型 A: 手写 LU 分解逻辑回归 =================
    print("\n" + "-"*40)
    print("[模型 A] 数值计算核心模型: LU分解逻辑回归")
    print("-"*40)
    print("原理: 利用 Newton-Raphson 法将最优化转化为线性方程组求解")
    print("      并使用 Partial Pivoting LU 分解加速求解 Hessian 矩阵")
    
    start_time = time.time()
    
    # 多分类策略: One-vs-Rest
    classes = np.unique(y)
    lu_models = {}
    
    for cls in classes:
        print(f"  -> 训练类别 {int(cls)} vs 其他...")
        y_binary = (y_train == cls).astype(int)
        model = ManualLogisticRegression(max_iter=15)
        model.fit(X_train_std, y_binary)
        lu_models[cls] = model
        
    # 预测
    y_pred_lu = []
    for i in range(len(X_test_std)):
        sample = X_test_std[i:i+1]
        # 获取所有类别的概率
        probs = {c: lu_models[c].predict_proba(sample)[0] for c in classes}
        # 取最大概率对应的类别
        y_pred_lu.append(max(probs, key=probs.get))
        
    time_lu = time.time() - start_time
    acc_lu = accuracy_score(y_test, y_pred_lu)
    results['LU-LogReg'] = {'acc': acc_lu, 'time': time_lu}
    print(f"结果: 准确率 = {acc_lu*100:.2f}%, 耗时 = {time_lu:.2f}s")
    
    # ================= 模型 B: 随机森林 (机器学习) =================
    print("\n" + "-"*40)
    print("[模型 B] 机器学习对比: 随机森林 (Random Forest)")
    print("-"*40)
    print("优势: 适应非线性，参数量小，推理快")
    
    start_time = time.time()
    # 限制树的数量和深度，保证轻量化
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train) # 树模型不需要标准化
    y_pred_rf = rf_model.predict(X_test)
    
    time_rf = time.time() - start_time
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results['RandomForest'] = {'acc': acc_rf, 'time': time_rf}
    print(f"结果: 准确率 = {acc_rf*100:.2f}%, 耗时 = {time_rf:.2f}s")
    
    # ================= 模型 C: 神经网络 (深度学习) =================
    print("\n" + "-"*40)
    print("[模型 C] 深度学习对比: 简易神经网络 (MLP)")
    print("-"*40)
    print("结构: 全连接层，模拟深度学习特征提取能力")
    
    start_time = time.time()
    # 使用 MLP 模拟轻量级深度学习 (无需安装 TensorFlow/PyTorch 即可运行)
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    mlp_model.fit(X_train_std, y_train)
    y_pred_mlp = mlp_model.predict(X_test_std)
    
    time_mlp = time.time() - start_time
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    results['NeuralNet'] = {'acc': acc_mlp, 'time': time_mlp}
    print(f"结果: 准确率 = {acc_mlp*100:.2f}%, 耗时 = {time_mlp:.2f}s")
    
    # ================= 4. 结果可视化与保存 =================
    print("\n正在生成对比报告...")
    
    # 保存训练日志和模型表现
    report = {
        'model_performance': results,
        'dataset_info': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train.shape[1]
        }
    }
    with open('model_training_log.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    # 绘制对比图
    plt.figure(figsize=(12, 5))
    
    # 子图1: 准确率
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    accs = [results[n]['acc']*100 for n in names]
    colors = ['#4c72b0', '#55a868', '#c44e52']
    bars = plt.bar(names, accs, color=colors)
    plt.ylim(0, 100)
    plt.ylabel('准确率 (%)')
    plt.title('各模型准确率对比')
    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, 
                 f'{bar.get_height():.1f}%', ha='center')
        
    # 子图2: 耗时 (对数坐标，因为手动LU可能慢一些)
    plt.subplot(1, 2, 2)
    times = [results[n]['time'] for n in names]
    plt.bar(names, times, color='gray', alpha=0.7)
    plt.ylabel('训练耗时 (秒)')
    plt.title('模型训练效率对比')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison_stage4.png')
    print("对比图已保存: model_comparison_stage4.png")
    print("训练日志已保存: model_training_log.json")
    
    print("\n" + "="*60)
    print("第四阶段完成！")
    print(f"推荐最佳模型: {max(results, key=lambda x: results[x]['acc'])}")
    print("="*60)

if __name__ == "__main__":
    run_stage_4()
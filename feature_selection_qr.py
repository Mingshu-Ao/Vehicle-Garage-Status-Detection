# feature_selection_qr.py
# 第三阶段：特征分析与筛选 (基于 QR 分解)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
except:
    pass

class NumericalAlgo:
    """数值计算核心算法库"""
    
    @staticmethod
    def modified_gram_schmidt(A):
        """
        手动实现修正的格拉姆-施密特正交化 (Modified Gram-Schmidt)
        用于计算 QR 分解: A = Q * R
        
        参数:
            A: m x n 矩阵
        返回:
            Q: m x n 正交矩阵
            R: n x n 上三角矩阵
        """
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        # 复制 A 避免修改原数据，使用 float64 保证精度
        V = A.astype(np.float64).copy()
        
        print(f"   [算法] 开始 QR 分解 (矩阵大小 {m}x{n})...")
        
        for i in range(n):
            # 1. 计算 R[i, i] (L2 范数)
            R[i, i] = np.linalg.norm(V[:, i])
            
            # 防止除以零
            if R[i, i] < 1e-10:
                print(f"     警告: 第 {i} 列几乎线性相关，R[{i},{i}] ≈ 0")
                continue
                
            # 2. 计算 Q[:, i] (归一化)
            Q[:, i] = V[:, i] / R[i, i]
            
            # 3. 正交化后续列 (移除当前方向的分量)
            # 修正 Gram-Schmidt 的关键：立即更新剩余的向量
            for j in range(i + 1, n):
                R[i, j] = np.dot(Q[:, i], V[:, j])
                V[:, j] = V[:, j] - R[i, j] * Q[:, i]
            
            # 进度条
            if n > 10 and i % int(n/5) == 0:
                print(f"     进度: {i+1}/{n} 列...")
                
        return Q, R

class FeatureSelector:
    def __init__(self, input_file='svd_processed_train.csv'):
        self.input_file = input_file
        self.X = None
        self.y = None
        self.feature_names = None
        self.selected_indices = None
        
    def load_data(self):
        """加载阶段2处理后的数据"""
        print("="*60)
        print("1. 加载 SVD 处理后的数据")
        print("="*60)
        
        if not os.path.exists(self.input_file):
            print(f"错误: 找不到文件 {self.input_file}")
            print("请确保已成功运行阶段2的代码")
            return False
            
        df = pd.read_csv(self.input_file)
        print(f"加载文件: {self.input_file}")
        print(f"数据形状: {df.shape}")
        
        if 'labelArea' not in df.columns:
            print("错误: 数据中缺少 labelArea 列")
            return False
            
        self.y = df['labelArea'].values
        self.X = df.drop(columns=['labelArea']).values
        self.feature_names = df.drop(columns=['labelArea']).columns.tolist()
        
        print(f"特征矩阵 X: {self.X.shape}")
        print(f"标签向量 y: {self.y.shape}")
        return True

    def analyze_correlation(self):
        """特征与标签的相关性分析"""
        print("\n" + "="*60)
        print("2. 特征-标签相关性分析")
        print("="*60)
        
        n_features = self.X.shape[1]
        correlations = []
        
        print("计算每个特征与标签的 Pearson 相关系数...")
        for i in range(n_features):
            # 计算特征列与标签列的相关系数
            # corrcoef 返回矩阵 [[1, r], [r, 1]]，取 [0,1]
            corr = np.corrcoef(self.X[:, i], self.y)[0, 1]
            correlations.append(abs(corr)) # 取绝对值，关注相关程度
            
        # 转换为 DataFrame 方便展示
        corr_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Correlation': correlations
        })
        
        # 排序
        corr_df = corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
        
        print("\n与标签关联度最高的 Top 10 特征:")
        print(corr_df.head(10))
        
        self.corr_df = corr_df
        return corr_df

    def perform_qr_selection(self, retention_ratio=0.8):
        """执行 QR 分解并筛选特征"""
        print("\n" + "="*60)
        print("3. 执行 QR 分解与特征筛选")
        print("="*60)
        
        # 1. 执行 QR 分解
        Q, R = NumericalAlgo.modified_gram_schmidt(self.X)
        
        # 2. 提取对角线元素 (特征重要性得分)
        r_diag = np.abs(np.diag(R))
        
        # 3. 生成评分表
        qr_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'R_diagonal': r_diag,
            'Original_Index': range(len(self.feature_names))
        })
        
        # 按重要性排序
        qr_scores = qr_scores.sort_values(by='R_diagonal', ascending=False)
        
        print("\nQR 分解 R 矩阵对角线得分 (Top 10):")
        print(qr_scores.head(10))
        
        # 4. 确定筛选数量 m
        # 策略：保留 R 对角线能量占比达到 retention_ratio 的特征
        # 或者简单的：如果特征少于20个，全部保留；否则保留 80%
        n_total = len(self.feature_names)
        
        # 这里演示：结合相关性和QR得分
        # 由于我们输入的是 SVD 后的 PC，它们本身已经是正交的，
        # 这里的 QR 分解更多是对数值稳定性的验证。
        # 对于分类任务，我们更倾向于保留 "与标签相关" 的特征。
        
        # 混合策略：
        # 1. 即使 PC 本身按方差排序，但未必按分类能力排序。
        # 2. 我们取 Correlation Top N 和 QR Stability Top N 的交集，或者加权。
        # 3. 简单起见，我们根据 QR 的 R值（代表方差/能量）截断尾部的噪音（如果 SVD 没切干净）
        
        # 自动选择 m: 找到 R_diag 衰减突变点，或者直接指定
        m = max(2, int(n_total * 0.8)) # 示例：再次压缩 20%
        
        print(f"\n筛选决策: 原始 {n_total} 个 -> 筛选后 {m} 个")
        
        # 获取 Top m 的特征索引
        top_features = qr_scores.head(m)['Feature'].tolist()
        self.selected_indices = qr_scores.head(m)['Original_Index'].values
        self.selected_features = top_features
        
        print(f"保留的特征列表 (前5个): {top_features[:5]} ...")
        
        return Q, R

    def save_and_visualize(self):
        """保存结果并可视化"""
        print("\n" + "="*60)
        print("4. 保存结果与可视化")
        print("="*60)
        
        if self.selected_indices is None:
            return
            
        # 1. 构建筛选后的数据集
        # 注意：这里需要按原始索引重新取列，保持顺序或按重要性排序
        # 为了后续模型训练方便，我们按重要性重新排序特征
        X_selected = self.X[:, self.selected_indices]
        
        # 创建 DataFrame
        selected_df = pd.DataFrame(X_selected, columns=self.selected_features)
        selected_df['labelArea'] = self.y
        
        # 保存
        out_path = 'feature_selected_train.csv'
        selected_df.to_csv(out_path, index=False)
        print(f"筛选后的数据集已保存: {out_path} ({selected_df.shape})")
        
        # 2. 可视化对比
        plt.figure(figsize=(12, 5))
        
        # 图1: R 对角线元素分布
        plt.subplot(1, 2, 1)
        # 从 qr_scores 里拿数据（需要重新访问，这里简化直接用刚才算的）
        Q, R = NumericalAlgo.modified_gram_schmidt(self.X) # 重新算一次或存起来
        r_diag = np.sort(np.abs(np.diag(R)))[::-1]
        
        plt.plot(r_diag, 'b-o', markersize=4)
        plt.axvline(x=len(self.selected_features), color='r', linestyle='--', label='Cutoff')
        plt.title('QR分解 - 特征重要性 (R对角线)')
        plt.xlabel('特征序号')
        plt.ylabel('|R_ii|')
        plt.legend()
        plt.grid(True)
        
        # 图2: 特征与标签相关性分布
        plt.subplot(1, 2, 2)
        corrs = self.corr_df['Correlation'].values
        plt.bar(range(len(corrs)), corrs, color='skyblue')
        plt.title('特征-标签相关性分布 (已排序)')
        plt.xlabel('特征排名')
        plt.ylabel('绝对相关系数')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('feature_selection_analysis.png')
        print("分析图表已保存: feature_selection_analysis.png")

if __name__ == "__main__":
    # 实例化并运行
    selector = FeatureSelector()
    
    if selector.load_data():
        # 1. 分析相关性
        selector.analyze_correlation()
        
        # 2. QR 分解筛选
        selector.perform_qr_selection()
        
        # 3. 保存
        selector.save_and_visualize()
        
        print("\n" + "="*60)
        print("第三阶段完成！")
        print("准备进入阶段 4：模型构建 (LU分解 + 逻辑回归)")
        print("="*60)
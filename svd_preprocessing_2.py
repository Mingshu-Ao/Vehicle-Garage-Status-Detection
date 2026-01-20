# svd_preprocessing.py
# 终极修复版：SVD预处理完整脚本
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import sys
import matplotlib.font_manager as fm

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 1. 字体与绘图设置 =================
def setup_plotting():
    """配置绘图风格和中文字体"""
    sns.set_style("whitegrid")
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 尝试设置中文字体
    font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    font_found = False
    
    # 1. 尝试系统字体名
    for name in font_names:
        try:
            plt.rcParams['font.sans-serif'] = [name]
            # 简单测试
            fig = plt.figure()
            plt.close(fig)
            font_found = True
            print(f"成功设置绘图字体: {name}")
            break
        except:
            continue
            
    # 2. 如果失败，尝试强制加载文件（针对Windows）
    if not font_found and sys.platform.startswith('win'):
        font_path = 'C:\\Windows\\Fonts\\simhei.ttf'
        if os.path.exists(font_path):
            my_font = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [my_font.get_name()]
            print(f"强制加载字体文件: {font_path}")

setup_plotting()

# ================= 2. 核心算法类：ManualSVD =================
class ManualSVD:
    """手动实现SVD分解的核心算法"""
    
    @staticmethod
    def standardize_data(X):
        """
        标准化数据 (Z-score)，并处理除以零的情况
        返回: X_std (标准化矩阵), mean_vec, std_vec
        """
        # 1. 检查输入是否含有NaN/Inf
        X_clean = np.copy(X)
        if np.isnan(X_clean).any() or np.isinf(X_clean).any():
            # 将Inf转为NaN，然后用列均值填充
            X_clean[np.isinf(X_clean)] = np.nan
            col_means = np.nanmean(X_clean, axis=0)
            # 找到NaN的位置
            inds = np.where(np.isnan(X_clean))
            # 填充
            X_clean[inds] = np.take(col_means, inds[1])
            
        # 2. 计算均值和标准差
        mean_vec = np.mean(X_clean, axis=0)
        std_vec = np.std(X_clean, axis=0)
        
        # 3. 处理标准差为0的情况（防止除以0）
        # 如果某列方差为0，说明是常量列，标准化后设为0
        zero_std_mask = (std_vec == 0)
        if np.any(zero_std_mask):
            std_vec[zero_std_mask] = 1.0  # 设为1避免除零错误
            
        # 4. 执行标准化
        X_std = (X_clean - mean_vec) / std_vec
        
        # 对于原本std为0的列，分子也是0，所以结果是0，符合预期
        return X_std, mean_vec, std_vec
    
    @staticmethod
    def power_method(A, max_iter=100, tol=1e-5):
        """幂方法计算最大特征值和特征向量（加速版）"""
        n = A.shape[0]
        # 随机初始化
        x = np.random.randn(n)
        if np.linalg.norm(x) == 0:
            x = np.ones(n)
        x = x / np.linalg.norm(x)
        
        eigenvalue = 0
        
        for _ in range(max_iter):
            x_old = x.copy()
            
            # 矩阵乘法 Ax
            x = A @ x
            
            # 归一化
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10: # 防止数值下溢
                break
                
            x = x / norm_x
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < tol:
                break
        
        # Rayleigh Quotient 计算特征值
        eigenvalue = x.T @ A @ x
        return eigenvalue, x
    
    @staticmethod
    def manual_svd(X, n_components=None):
        """
        执行手动SVD流程
        1. 计算协方差矩阵
        2. 特征值分解
        3. 计算奇异值和向量
        """
        n_samples, n_features = X.shape
        if n_components is None:
            n_components = min(n_samples, n_features)
            
        # 1. 计算协方差矩阵 (X^T * X) / (n-1)
        # 注意：输入X必须已经是标准化过的
        print("   [算法] 计算协方差矩阵...")
        cov_matrix = (X.T @ X) / (n_samples - 1)
        
        # 2. 特征值分解 (使用 Deflation + Power Method)
        print("   [算法] 开始特征值分解 (这可能需要一点时间)...")
        eigenvalues = []
        eigenvectors = []
        A_curr = cov_matrix.copy()
        
        # 限制最大计算的主成分数，防止太慢
        # 如果特征很多，为了作业演示，可以只计算前100个或者直到方差贡献足够
        limit_components = min(n_components, 150) 
        
        for i in range(limit_components):
            if i % 10 == 0:
                print(f"     正在计算第 {i+1} 个主成分...")
                
            val, vec = ManualSVD.power_method(A_curr)
            
            eigenvalues.append(val)
            eigenvectors.append(vec)
            
            # 紧缩 (Deflation): A_new = A - lambda * v * v^T
            A_curr = A_curr - val * np.outer(vec, vec)
            
            # 如果特征值太小，提前停止
            if val < 1e-10:
                break
                
        eigenvalues = np.array(eigenvalues)
        V = np.column_stack(eigenvectors) # 右奇异向量 (转置前)
        
        # 3. 计算奇异值 S = sqrt(eigenvalues * (n-1))
        # 只有正特征值才有意义
        valid_idx = eigenvalues > 0
        eigenvalues = eigenvalues[valid_idx]
        V = V[:, valid_idx]
        
        S = np.sqrt(eigenvalues * (n_samples - 1))
        
        # 4. 计算左奇异向量 U = X * V / S
        # U = X @ V @ np.diag(1.0 / S)
        # 为了数值稳定性，分开计算
        U = X @ V
        U = U / S  # 广播除法
        
        return U, S, V.T

# ================= 3. 业务逻辑类：SVDPreprocessor =================
class SVDPreprocessor:
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.fitted = False
        
    def load_data(self):
        """加载数据并进行强力清洗"""
        base_dir = "D:/bupt/code/python/数值计算期末作业数据" # 请确认此路径
        file_path = os.path.join(base_dir, "train_data.csv")
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            # 尝试当前目录
            file_path = "train_data.csv"
            if not os.path.exists(file_path):
                return None, None
        
        print(f"加载文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 1. 剔除绝对不需要的列
        drop_cols = ['time', 'group_name', 'light_is_daytime']
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        if cols_to_drop:
            print(f"剔除ID/时间列: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            
        # 2. 提取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_num = df[numeric_cols]
        
        # 3. 处理 Inf 和 NaN
        print("处理缺失值和无穷值...")
        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        df_num = df_num.fillna(df_num.mean()).fillna(0)
        
        # 4. 剔除方差为 0 的列 (常量列) - 非常重要！
        print("剔除常量列 (方差为0)...")
        variances = df_num.var()
        constant_cols = variances[variances == 0].index
        if len(constant_cols) > 0:
            print(f"  发现 {len(constant_cols)} 个常量列，已剔除。")
            df_num = df_num.drop(columns=constant_cols)
            
        # 5. 分离 X 和 y
        if 'labelArea' in df_num.columns:
            y = df_num['labelArea'].values
            X_df = df_num.drop(columns=['labelArea'])
            X = X_df.values
            self.feature_names = X_df.columns.tolist()
            return X, y
        else:
            print("错误: 未找到 labelArea 列")
            return None, None

    def fit_transform(self, X):
        """执行SVD流程"""
        print(f"\n原始特征维度: {X.shape}")
        
        # 1. 标准化
        print("1. 执行标准化 (Z-score)...")
        X_std, self.mean_, self.std_ = ManualSVD.standardize_data(X)
        
        # 再次检查标准化后是否有 NaN
        if np.isnan(X_std).any():
            print("错误: 标准化后仍存在 NaN，强行置0")
            X_std = np.nan_to_num(X_std)
            
        # 2. SVD分解
        print("2. 执行手动SVD分解...")
        self.U, self.S, self.Vt = ManualSVD.manual_svd(X_std)
        
        # 3. 选择 K 值
        total_variance = np.sum(self.S ** 2)
        explained_variance_ratio = (self.S ** 2) / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # 找到满足阈值的 k
        k = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        self.k = k
        
        print(f"\n分解完成:")
        print(f"   保留主成分数 k: {self.k} / {len(self.S)}")
        print(f"   累计方差贡献率: {cumulative_variance[k-1]*100:.2f}%")
        
        # 4. 转换数据
        print(f"3. 转换数据 (降维到 {k} 维)...")
        # 投影公式: X_new = X_std * V_k
        V_k = self.Vt.T[:, :k]
        X_new = X_std @ V_k
        
        self.cumulative_variance = cumulative_variance
        self.explained_variance_ratio = explained_variance_ratio
        
        return X_new

    def visualize(self, X_new, save_dir='svd_results'):
        """可视化"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.figure(figsize=(15, 5))
        
        # 1. 碎石图
        plt.subplot(1, 3, 1)
        plt.plot(self.S[:50], 'bo-') # 只画前50个，避免太密集
        plt.title('奇异值 (Scree Plot, Top 50)')
        plt.ylabel('Singular Value')
        
        # 2. 累计方差
        plt.subplot(1, 3, 2)
        plt.plot(self.cumulative_variance * 100, 'g-')
        plt.axhline(y=self.variance_threshold*100, color='r', linestyle='--')
        plt.axvline(x=self.k, color='r', linestyle='--')
        plt.title('累计方差贡献率')
        plt.xlabel('k')
        plt.ylabel('Variance Ratio (%)')
        
        # 3. 2D 散点图
        plt.subplot(1, 3, 3)
        if X_new.shape[1] >= 2:
            plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.5, s=5)
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.title('前两个主成分投影')
        else:
            plt.text(0.5, 0.5, "降维后只有1维\n无法绘制散点图", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'svd_analysis.png'))
        plt.show()
        print(f"\n图表已保存到 {save_dir}/svd_analysis.png")

# ================= 4. 主程序入口 =================
if __name__ == "__main__":
    print(">>> 开始运行 SVD 数据预处理 (修复版) <<<")
    
    # 初始化
    processor = SVDPreprocessor(variance_threshold=0.95)
    
    # 1. 加载
    X, y = processor.load_data()
    
    if X is not None:
        # 2. 处理
        X_svd = processor.fit_transform(X)
        
        # 3. 保存结果
        output_df = pd.DataFrame(X_svd, columns=[f'PC{i+1}' for i in range(X_svd.shape[1])])
        output_df['labelArea'] = y
        output_df.to_csv('svd_processed_train.csv', index=False)
        print(f"结果已保存: svd_processed_train.csv ({output_df.shape})")
        
        # 4. 可视化
        processor.visualize(X_svd)
        
        print("\n>>> 处理成功完成 <<<")
    else:
        print(">>> 处理失败：无法加载数据 <<<")
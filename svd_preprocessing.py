# svd_preprocessing_fixed.py
# 修正版SVD预处理代码
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import sys

warnings.filterwarnings('ignore')

# 修复中文字体显示问题
import matplotlib.font_manager as fm  # 记得在文件最开头加这一行

def setup_chinese_font():
    """强制加载字体文件模式"""
    sns.set_style("whitegrid")
    plt.rcParams['axes.unicode_minus'] = False
    
    # Windows 字体绝对路径
    font_path = 'C:\\Windows\\Fonts\\simhei.ttf'  # 黑体
    # 如果想用微软雅黑，路径通常是 'C:\\Windows\\Fonts\\msyh.ttc'
    
    if os.path.exists(font_path):
        my_font = fm.FontProperties(fname=font_path)
        # 将这个字体对象应用到全局默认字体
        plt.rcParams['font.sans-serif'] = [my_font.get_name()]
        print(f"已强制加载字体文件: {font_path}")
    else:
        print(f"未找到字体文件: {font_path}，将尝试使用系统默认设置")
        plt.rcParams['font.sans-serif'] = ['SimHei']
# 调用字体设置
setup_chinese_font()

class ManualSVD:
    """手动实现SVD分解"""
    
    @staticmethod
    def standardize_data(X):
        """标准化数据，处理NaN和无穷值"""
        # 检查并处理NaN和无穷值
        X_clean = np.copy(X)
        
        # 用列均值替换NaN
        col_means = np.nanmean(X_clean, axis=0)
        nan_indices = np.isnan(X_clean)
        X_clean[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
        
        # 处理无穷值
        inf_indices = np.isinf(X_clean)
        X_clean[inf_indices] = np.nan
        col_means = np.nanmean(X_clean, axis=0)
        X_clean[inf_indices] = np.take(col_means, np.where(inf_indices)[1])
        
        # 标准化
        mean_vec = np.mean(X_clean, axis=0)
        std_vec = np.std(X_clean, axis=0)
        std_vec[std_vec == 0] = 1.0  # 避免除零
        
        X_std = (X_clean - mean_vec) / std_vec
        return X_std, mean_vec, std_vec
    
    @staticmethod
    def power_method(A, max_iter=1000, tol=1e-10):
        """幂方法计算最大特征值和特征向量"""
        n = A.shape[0]
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        for i in range(max_iter):
            x_old = x.copy()
            x = A @ x
            x = x / np.linalg.norm(x)
            if np.linalg.norm(x - x_old) < tol:
                break
        
        eigenvalue = (x.T @ A @ x) / (x.T @ x)
        return eigenvalue, x
    
    @staticmethod
    def deflation(A, eigenvalue, eigenvector):
        """紧缩技术"""
        return A - eigenvalue * np.outer(eigenvector, eigenvector)
    
    @staticmethod
    def eigendecomposition(A, n_components=None):
        """手动特征值分解"""
        n = A.shape[0]
        if n_components is None or n_components > n:
            n_components = n
        
        eigenvalues = []
        eigenvectors = []
        A_current = A.copy()
        
        for i in range(n_components):
            eigenvalue, eigenvector = ManualSVD.power_method(A_current)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            
            if i < n_components - 1:
                A_current = ManualSVD.deflation(A_current, eigenvalue, eigenvector)
        
        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.column_stack(eigenvectors)
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def manual_svd(X, n_components=None):
        """手动SVD分解"""
        n_samples = X.shape[0]
        
        # 计算协方差矩阵
        cov_matrix = (X.T @ X) / (n_samples - 1)
        
        # 特征值分解
        eigenvalues, V = ManualSVD.eigendecomposition(cov_matrix, n_components)
        
        # 奇异值
        S = np.sqrt(eigenvalues * (n_samples - 1))
        
        # 左奇异向量
        S_nonzero = S.copy()
        S_nonzero[S_nonzero == 0] = 1.0
        U = X @ V @ np.diag(1.0 / S_nonzero)
        
        # 标准化U
        for i in range(U.shape[1]):
            norm = np.linalg.norm(U[:, i])
            if norm > 0:
                U[:, i] = U[:, i] / norm
        
        return U, S, V.T
    
    @staticmethod
    def select_k_by_variance(S, variance_threshold=0.95):
        """根据累计方差选择k值"""
        variance = S ** 2
        total_variance = np.sum(variance)
        
        # 检查total_variance是否为0
        if total_variance == 0:
            return 1, np.zeros_like(S), np.zeros_like(S)
        
        explained_variance_ratio = variance / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        k = np.argmax(cumulative_variance >= variance_threshold) + 1
        k = max(1, min(k, len(S)))  # 确保k在合理范围内
        
        return k, explained_variance_ratio, cumulative_variance


class SVDPreprocessor:
    """SVD数据预处理器"""
    
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.mean_vec = None
        self.std_vec = None
        self.U = None
        self.S = None
        self.Vt = None
        self.k = None
        self.fitted = False
    
    def clean_data(self, X):
        """清理数据，移除NaN和无穷值"""
        X_clean = np.copy(X)
        
        # 移除全为0的列
        zero_cols = np.all(X_clean == 0, axis=0)
        if np.any(zero_cols):
            print(f"移除 {np.sum(zero_cols)} 个全零列")
            X_clean = X_clean[:, ~zero_cols]
        
        # 用列均值替换NaN
        col_means = np.nanmean(X_clean, axis=0)
        nan_indices = np.isnan(X_clean)
        X_clean[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
        
        # 处理无穷值
        inf_indices = np.isinf(X_clean)
        X_clean[inf_indices] = 0
        
        return X_clean
    
    def fit(self, X):
        """拟合SVD模型"""
        print("1. 标准化数据...")
        
        # 清理数据
        X_clean = self.clean_data(X)
        
        X_std, self.mean_vec, self.std_vec = ManualSVD.standardize_data(X_clean)
        
        print("2. 执行手动SVD分解...")
        self.U, self.S, self.Vt = ManualSVD.manual_svd(X_std)
        
        print("3. 选择最佳k值...")
        self.k, self.explained_variance_ratio, self.cumulative_variance = \
            ManualSVD.select_k_by_variance(self.S, self.variance_threshold)
        
        print(f"   原始特征维度: {X.shape[1]}")
        print(f"   选择的k值: {self.k} (保留{self.variance_threshold*100:.1f}%方差)")
        print(f"   降维比例: {(1 - self.k/X.shape[1])*100:.1f}%")
        
        self.fitted = True
        return self
    
    def transform(self, X, k=None):
        """转换数据（降维）"""
        if not self.fitted:
            raise ValueError("请先调用fit方法")
        
        if k is None:
            k = self.k
        
        # 清理和标准化
        X_clean = self.clean_data(X)
        X_std = (X_clean - self.mean_vec) / self.std_vec
        
        # 投影到主成分
        V = self.Vt.T
        X_svd = X_std @ V[:, :k]
        
        return X_svd
    
    def fit_transform(self, X, k=None):
        """拟合并转换数据"""
        self.fit(X)
        return self.transform(X, k)
    
    def analyze_effect(self, X_original, X_processed):
        """分析预处理效果（修复版：防止NaN卡死）"""
        print("\n" + "="*60)
        print("SVD预处理效果分析")
        print("="*60)
        
        # 1. 方差分析
        # 注意：使用累计方差贡献率作为最准确的指标
        retained_ratio = self.cumulative_variance[self.k-1]
        
        print(f"1. 方差分析:")
        print(f"   降维后特征数: {X_processed.shape[1]}")
        print(f"   保留方差比例: {retained_ratio*100:.2f}%")
        
        # 2. 相关性分析 (增加安全性检查)
        print(f"\n2. 相关性分析:")
        
        try:
            # 为了速度，只取前 500 个样本计算相关性
            n_samples = min(500, X_original.shape[0])
            
            # 计算原始相关性（处理 NaN）
            X_orig_sample = X_original[:n_samples]
            # 剔除方差为0的列，防止除以0产生NaN
            valid_cols = np.var(X_orig_sample, axis=0) > 1e-6
            
            if np.sum(valid_cols) > 1:
                X_orig_clean = X_orig_sample[:, valid_cols]
                corr_orig = np.corrcoef(X_orig_clean, rowvar=False)
                # 忽略对角线
                np.fill_diagonal(corr_orig, np.nan)
                avg_corr_orig = np.nanmean(np.abs(corr_orig))
                print(f"   原始数据平均绝对相关性: {avg_corr_orig:.4f}")
            else:
                avg_corr_orig = 0
                print("   原始数据有效列不足，跳过")

            # 计算处理后相关性
            if X_processed.shape[1] > 1:
                X_proc_sample = X_processed[:n_samples]
                corr_proc = np.corrcoef(X_proc_sample, rowvar=False)
                np.fill_diagonal(corr_proc, np.nan)
                avg_corr_proc = np.nanmean(np.abs(corr_proc))
                print(f"   处理后数据平均绝对相关性: {avg_corr_proc:.4f}")
                
                if avg_corr_orig > 0:
                    reduction = (1 - avg_corr_proc/avg_corr_orig) * 100
                    print(f"   相关性降低比例: {reduction:.1f}%")
            else:
                print(f"   处理后仅剩1维，无需计算相关性")

        except Exception as e:
            print(f"   警告：计算相关性时遇到错误 (已跳过): {e}")

        return {
            'retained_variance_ratio': retained_ratio
        }
    
    def visualize(self, X_processed, save_dir='svd_results'):
        """可视化结果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 奇异值碎石图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(self.S) + 1), self.S, 'bo-', linewidth=2)
        plt.axvline(x=self.k, color='r', linestyle='--', label=f'k={self.k}')
        plt.xlabel('Principal Component Index', fontsize=12) # 主成分序号
        plt.ylabel('Singular Value', fontsize=12)            # 奇异值
        plt.title('Scree Plot', fontsize=14, fontweight='bold') # 奇异值碎石图
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 累计方差图
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(self.cumulative_variance) + 1), 
                self.cumulative_variance * 100, 'go-', linewidth=2)
        plt.axhline(y=self.variance_threshold * 100, color='r', linestyle='--')
        plt.axvline(x=self.k, color='r', linestyle='--')
        plt.xlabel('Number of Components', fontsize=12)      # 主成分数量
        plt.ylabel('Cumulative Variance Ratio (%)', fontsize=12) # 累计方差解释比例
        plt.title('Cumulative Variance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加文本标注
        if self.k <= len(self.cumulative_variance):
            cum_var = self.cumulative_variance[self.k-1] * 100
            plt.text(self.k + 0.5, self.variance_threshold * 100 - 5, 
                    f'k={self.k}\n{cum_var:.1f}%', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 方差解释比例（前10个）
        plt.subplot(1, 3, 3)
        n_to_show = min(10, len(self.explained_variance_ratio))
        plt.bar(range(1, n_to_show + 1), 
               self.explained_variance_ratio[:n_to_show] * 100, 
               color='skyblue', alpha=0.7)
        plt.xlabel('主成分序号', fontsize=12)
        plt.ylabel('方差解释比例 (%)', fontsize=12)
        plt.title(f'前{n_to_show}个主成分的方差解释比例', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('SVD预处理分析结果', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(save_dir, 'svd_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到: {save_path}")
        
        # 如果降维后维度大于1，绘制散点图
        if X_processed.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(X_processed[:, 0], X_processed[:, 1], 
                       alpha=0.5, s=10, c='blue', edgecolors='none')
            plt.xlabel('第一主成分', fontsize=12)
            plt.ylabel('第二主成分', fontsize=12)
            plt.title('数据在前两个主成分上的投影', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            scatter_path = os.path.join(save_dir, 'svd_scatter.png')
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"散点图已保存到: {scatter_path}")


def load_and_prepare_data():
    """加载并准备数据（终极修复版：自动剔除异常高方差列）"""
    print("="*60)
    print("加载数据")
    print("="*60)
    
    # 路径配置
    base_dir = "D:/bupt/code/python/数值计算期末作业数据"
    file_path = os.path.join(base_dir, "train_data.csv")
    test_dir = os.path.join(base_dir, "test")
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return None, None, None, None
    
    try:
        print(f"加载文件: {file_path}")
        train_df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 1. 基础清理：剔除已知的非数值/干扰列
        cols_to_drop = ['time', 'group_name', 'light_is_daytime']
        existing_cols = [c for c in cols_to_drop if c in train_df.columns]
        if existing_cols:
            print(f"剔除基础干扰列: {existing_cols}")
            train_df = train_df.drop(columns=existing_cols)
        
        # 2. 强力数据清洗 (处理 NaN)
        print("执行 NaN 清洗...")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        # 先用列均值填充
        train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
        # 剩下的（全NaN列）填0
        train_df = train_df.fillna(0)
        
        # =================【新增核心逻辑】=================
        # 3. 自动剔除方差极大的异常列（这是导致 k=1 的元凶）
        # 计算每一列的方差
        print("正在检查特征方差...")
        variances = train_df[numeric_cols].var()
        
        # 设定阈值：例如 1e9 (十亿)，正常传感器数据不会超过这个方差
        # 你的数据里有些列方差达到了 1e17，必须删掉
        variance_threshold = 1e9 
        high_variance_cols = variances[variances > variance_threshold].index.tolist()
        
        if high_variance_cols:
            print(f"警告: 发现 {len(high_variance_cols)} 个方差过大的异常列 (Variance > {variance_threshold:.0e})")
            print(f"示例: {high_variance_cols[:3]}...")
            print("正在自动剔除这些异常列...")
            train_df = train_df.drop(columns=high_variance_cols)
        # =================================================
            
        print(f"数据清洗完成，最终形状: {train_df.shape}")
        
        # 检查非数值列 (Double check)
        non_numeric_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            train_df = train_df.drop(columns=non_numeric_cols)
            
        if 'labelArea' in train_df.columns:
            y = train_df['labelArea']
            X = train_df.drop('labelArea', axis=1)
            
            # 再次确认数据质量
            if np.isinf(X.values).any() or np.isnan(X.values).any():
                print("警告: 数据中仍含有 Inf 或 NaN，执行强制替换")
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return X.values, y.values, [], []
        else:
            print("错误: 数据中没有 labelArea 列")
            return None, None, None, None
            
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def run_svd_pipeline():
    """运行完整的SVD预处理流程"""
    print("="*60)
    print("第二阶段：SVD数据预处理")
    print("="*60)
    
    # 1. 加载数据
    X, y, test_dfs, test_filenames = load_and_prepare_data()
    if X is None:
        print("数据加载失败，终止流程")
        return None
    
    print(f"\n原始数据统计:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  数据范围: [{X.min():.2e}, {X.max():.2e}]")
    print(f"  数据均值: {X.mean():.2e}")
    print(f"  数据标准差: {X.std():.2e}")
    
    # 2. 创建SVD预处理器
    svd = SVDPreprocessor(variance_threshold=0.95)
    
    # 3. 拟合并转换训练集
    print("\n" + "="*60)
    print("进行SVD预处理")
    print("="*60)
    
    try:
        X_processed = svd.fit_transform(X)
    except Exception as e:
        print(f"SVD处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. 分析效果
    analysis_results = svd.analyze_effect(X, X_processed)
    
    # 5. 可视化
    try:
        svd.visualize(X_processed)
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 6. 转换测试集
    X_test_processed_list = []
    if test_dfs:
        print("\n" + "="*60)
        print("处理测试集")
        print("="*60)
        for i, test_df in enumerate(test_dfs):
            try:
                X_test = test_df.values
                X_test_processed = svd.transform(X_test)
                X_test_processed_list.append(X_test_processed)
                print(f"测试文件 {test_filenames[i]}: {X_test.shape} -> {X_test_processed.shape}")
            except Exception as e:
                print(f"处理测试文件 {test_filenames[i]} 失败: {e}")
    
    # 7. 保存结果
    print("\n" + "="*60)
    print("保存处理结果")
    print("="*60)
    
    # 保存训练集
    if y is not None:
        n_components = X_processed.shape[1]
        feature_columns = [f'PC{i+1}' for i in range(n_components)]
        train_processed_df = pd.DataFrame(X_processed, columns=feature_columns)
        train_processed_df['labelArea'] = y
        train_processed_df.to_csv('svd_processed_train.csv', index=False, encoding='utf-8-sig')
        print(f"保存训练集: svd_processed_train.csv ({train_processed_df.shape})")
    
    # 保存测试集
    if X_test_processed_list:
        test_output_dir = 'svd_processed_test'
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        
        for i, (X_test_processed, filename) in enumerate(zip(X_test_processed_list, test_filenames)):
            n_components = X_test_processed.shape[1]
            feature_columns = [f'PC{i+1}' for i in range(n_components)]
            test_processed_df = pd.DataFrame(X_test_processed, columns=feature_columns)
            output_path = os.path.join(test_output_dir, f'svd_{filename}')
            test_processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"保存测试集: {output_path}")
    
    # 8. 生成报告
    print("\n" + "="*60)
    print("生成处理报告")
    print("="*60)
    
    report = f"""SVD预处理报告
===============
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 处理参数
-----------
方差阈值: {svd.variance_threshold}
选择的k值: {svd.k}
原始特征维度: {X.shape[1]}
降维后维度: {X_processed.shape[1]}
降维比例: {(1 - svd.k/X.shape[1])*100:.1f}%

2. 方差分析
-----------
原始数据总方差: {analysis_results['total_variance_original']:.4f}
处理后数据总方差: {analysis_results['total_variance_processed']:.4f}
保留方差比例: {analysis_results['retained_variance_ratio']*100:.2f}%

3. 奇异值分析
-------------
总奇异值数量: {len(svd.S)}
最大奇异值: {svd.S[0]:.4f}
最小奇异值: {svd.S[-1]:.4f}
奇异值衰减比率: {svd.S[-1]/svd.S[0]*100:.2f}%

4. 主成分方差解释
----------------"""
    
    for i in range(min(10, len(svd.explained_variance_ratio))):
        ratio = svd.explained_variance_ratio[i] * 100
        cum_ratio = svd.cumulative_variance[i] * 100
        report += f"\nPC{i+1}: {ratio:.2f}% (累计: {cum_ratio:.2f}%)"
    
    report += f"""

5. 结果分析
-----------
✓ 成功完成SVD预处理
✓ 降维效果: {X.shape[1]}维 → {X_processed.shape[1]}维
✓ 保留方差比例: {analysis_results['retained_variance_ratio']*100:.2f}%
✓ k={svd.k}时达到{svd.variance_threshold*100:.1f}%方差阈值

6. 输出文件
-----------
svd_processed_train.csv - 处理后的训练集
svd_results/ - 分析图表
svd_preprocessing_report.txt - 本报告

完成！
==============="""
    
    with open('svd_preprocessing_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("报告摘要:")
    print("-"*40)
    print(report[:500] + "..." if len(report) > 500 else report)
    print(f"\n详细报告已保存到: svd_preprocessing_report.txt")
    
    print("\n" + "="*60)
    print("第二阶段完成！")
    print("="*60)
    
    return {
        'svd': svd,
        'X_original': X,
        'X_processed': X_processed,
        'y': y,
        'analysis_results': analysis_results
    }


if __name__ == "__main__":
    # 确保只运行一次
    if 'run_svd_pipeline' not in globals():
        globals()['run_svd_pipeline'] = run_svd_pipeline
    
    # 运行SVD预处理流程
    print("开始运行SVD预处理...")
    results = run_svd_pipeline()
    
    if results:
        print("\n处理完成！")
        print("="*60)
        print("结果摘要:")
        print(f"1. 原始维度: {results['X_original'].shape[1]}维")
        print(f"2. 处理后维度: {results['X_processed'].shape[1]}维")
        print(f"3. 降维比例: {(1 - results['X_processed'].shape[1]/results['X_original'].shape[1])*100:.1f}%")
        print(f"4. 保留方差: {results['analysis_results']['retained_variance_ratio']*100:.2f}%")
        print(f"5. 输出文件: svd_processed_train.csv")
        
        print("\n" + "="*60)
        print("准备进入第三阶段: 特征分析与筛选")
        print("="*60)
    else:
        print("\n处理失败，请检查错误信息。")
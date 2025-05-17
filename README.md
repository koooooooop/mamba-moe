# Mamba优化工作负载预测实验指南

## 项目概述

本项目实现了四种基于Mamba状态空间模型的工作负载预测优化方法，用于云计算和微服务架构中的资源使用预测。通过对比分析不同模型在各类工作负载场景下的表现，可以为不同应用场景选择最优预测模型。

## 模型方案

1. **选择性MambaMOE (Selective_MambaMOE.py)**
   - 基于模式的专家选择，动态路由输入到最相关专家
   - 选择性计算机制减少不必要计算
   - 自适应门控专注于特定工作负载模式
   - 适用于：具有明显不同模式的异构工作负载

2. **层次化Mamba (Hierarchical_Mamba.py)**
   - 多尺度处理捕获不同时间分辨率的模式
   - 三层抽象结构：局部、中程、全局
   - 自适应池化提取固定表示
   - 适用于：具有多尺度时间动态特性的复杂工作负载

3. **频域增强Mamba (FrequencyMamba.py)**
   - FFT特征提取捕获频域信息
   - 时域与频域双通道处理
   - 跨域融合整合不同特征
   - 适用于：具有强周期性组件的工作负载

4. **混合Mamba-Transformer (HybridMambaTransformer.py)**
   - 结合Mamba局部建模与Transformer全局依赖捕获
   - 门控集成机制智能融合两种结构
   - 位置编码保留序列顺序信息
   - 适用于：同时具有局部趋势和长距离依赖的工作负载

## 环境配置

### 依赖要求

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- scikit-learn
- matplotlib
- tqdm
- mamba-ssm

### 硬件要求

- CUDA兼容GPU (建议 ≥ 8GB 显存)
- RAM: ≥ 16GB

### 环境安装

安装基础依赖和Mamba状态空间模型库：

```
pip install torch numpy pandas scikit-learn matplotlib tqdm
pip install mamba-ssm
```

## 数据准备

确保数据集按照以下目录结构组织：

```
dataset/
├── prediction/
│   ├── pre-train/  - 预训练数据
│   │   ├── cluster_1.txt  - 聚类1的文件列表
│   │   ├── ...
│   │   └── cluster_12.txt - 聚类12的文件列表
│   ├── train/      - 训练数据
│   ├── test/       - 测试数据
├── model-para/     - 模型参数保存目录
├── output/         - 结果输出目录
```

对于聚类部分，确保`dataset/prediction/pre-train/`目录下有`cluster_1.txt`到`cluster_12.txt`文件，每个文件包含该聚类的工作负载CSV文件名列表。每个CSV文件应包含至少`cpu`,`memory`,`disk`三列资源使用数据。

## 实验流程

### 1. 批量运行所有模型

创建并运行批量实验脚本，该脚本将依次执行所有模型并收集性能指标：

```
python run_experiments.py
```

该脚本会运行所有模型并将结果保存至`dataset/output/model_comparison.csv`，同时生成对比图表。

### 2. 单独运行各模型

或者，您可以分别运行各个模型进行测试：

```
# 运行选择性MambaMOE模型
python Selective_MambaMOE.py

# 运行层次化Mamba模型
python Hierarchical_Mamba.py

# 运行频域增强Mamba模型
python FrequencyMamba.py

# 运行混合Mamba-Transformer模型
python HybridMambaTransformer.py

# 运行原始基准模型
python MOE_FLOW_MAMBA.py
python MOE_FLOW_FFT_LSTM.py
python Transformer.py
```

### 3. 详细性能分析

运行详细分析脚本，针对不同工作负载类型进行精细评估：

```
python analyze_performance.py
```

该脚本将分析各模型在不同工作负载类型（周期型、突发型、稳定型、趋势型）上的表现，并输出详细结果。

### 4. 生成实验报告

最后，生成包含所有结果和建议的HTML报告：

```
python generate_report.py
```

报告将保存至`dataset/output/report/experiment_report.html`，包含总体性能对比、性能提升分析、工作负载类型性能分析以及针对性建议。

## 评估指标

所有模型使用以下指标进行评估：

- **MSE (均方误差)**: 对大偏差更敏感
- **MAE (平均绝对误差)**: 整体误差的线性衡量
- **MAPE (平均绝对百分比误差)**: 反映相对误差水平
- **执行时间**: 模型预测的计算效率

## 工作负载类型分析

为全面评估模型性能，我们根据特性将工作负载分为几类：

1. **周期性工作负载**: 具有日常、周期性重复模式
2. **突发型工作负载**: 包含突然的资源使用峰值
3. **稳定型工作负载**: 变化较小，相对平稳
4. **趋势型工作负载**: 具有长期增长或下降趋势

每种模型在不同类型工作负载上的表现各异，详细结果将在分析报告中呈现。

## 使用建议

根据工作负载特性选择模型：

1. 对于周期性工作负载，推荐使用**FrequencyMamba**模型，其频域特征提取能力对此类数据最有效
2. 对于瞬时突发型工作负载，**SelectiveMambaMOE**模型专家机制能更好适应负载突变
3. 多尺度时间模式的复杂工作负载建议使用**HierarchicalMamba**模型
4. 对于同时具有长期依赖和短期模式的负载，**HybridMambaTransformer**可能是最佳选择

## 故障排除

1. **内存错误**: 如果遇到内存不足，尝试减小批量大小
2. **CUDA错误**: 确保CUDA版本与PyTorch和mamba-ssm兼容
3. **数据路径错误**: 确保文件路径正确，遵循指定的目录结构
4. **实验中断**: 每个模型会保存最佳检查点，可以从中断处继续训练

## 未来工作方向

1. **集成方法**: 组合多个模型的预测结果以提高整体性能
2. **迁移学习**: 在更大的数据集上预训练模型
3. **可解释性**: 增加模型解释机制以理解预测决策
4. **动态架构适应**: 运行时自动调整模型结构
5. **不确定性量化**: 提供带有置信区间的概率预测 
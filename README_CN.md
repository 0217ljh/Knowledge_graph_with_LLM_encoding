# 项目目录概述

本文档提供了项目目录及其功能的结构化概述。

---

## 目录
1. [cache_data](#cache_data)
2. [Configs](#configs)
3. [Data_process](#data_process)
4. [Database](#database)
5. [GNN_LLM_Spectrum](#gnn_llm_spectrum)
6. [Light](#light)
7. [Models](#models)
8. [saved_exp](#saved_exp)
9. [Spectrum simulation](#spectrum-simulation)
10. [task_data](#task_data)
11. [util](#util)
12. [wandb](#wandb)
13. [fs_train&generate](#fs_traingenerate)
14. [main](#main)
15. [zs_generate](#zs_generate)

---

## cache_data
存储不同任务的prompt图数据。
- **FS_Task**: 包含FS的prompt图。prompt图是通过融合知识节点图和光谱节点图，加入prompt信息并编码后生成的统一图结构。

## Configs
包含模型、任务和训练的配置。
- **data_config**: 数据配置。
- **default_config**: 默认模型配置。
- **task_config**: 任务相关配置。
- **overRide_task_config**: 可修改的任务和模型配置覆盖。

## Data_process
数据处理与融合模块，将原始数据转化为图结构数据。
- **data_collect/datatype/knowledge_map**: 处理知识图数据。
- **data_collect/datatype/spectrum**: 处理光谱图数据。
- **Merge_collect, Merge_process**: 融合图结构。

## Database
存储原始数据及标签。
- **FS_Database**: FS原始数据。
- **Functional group**: 官能团数据。
- **Knowledge graph**: OWL知识图谱。
- **Spectrum_database**: 原始光谱数据。
- **Task_set**: 不同任务的标签数据。

## GNN_LLM_Spectrum
存储模型权重。

## Light
与训练相关的模块。

## Models
模型相关模块。
- **LLM**: 包括编码器和模型定义。
- **nn**: 包括GNN和GAT实现。

## saved_exp
存储历史实验参数。

## Spectrum simulation
独立的光谱模拟与生成模块。

### 子模块
1. **data**: 基于QM9分子数据集（需解压），包含10万多个分子的相关物理信息。
2. **LLM**: 包含模型定义和编码器。
3. **code**:
   - **detanet_model**: 包含物理量预测与光谱模拟代码以及训练模块。
     - **detanet.py**: 核心模块，使用E3图神经网络进行物理量预测。
     - **spectra_simulator**: 光谱模拟模块，通过Detanet预测的物理量生成光谱。
     - **else**: 训练模块。
   - **trained_param**: 存储历史训练参数和模型权重。
   - **Dockerfile**: 模块镜像文件。
   - **main_train**: Detanet的训练主程序。
   - **main_calculate**: 使用训练好的Detanet进行物理量预测和光谱模拟的主程序。

## task_data
子任务的数据集构建模块。
- **edge**: 构建边任务的数据集。
- **graph**: 构建图任务的数据集。
  - **cls**: 子图分类。
  - **jdm**: 子图鉴定。
- **node**: 构建节点任务的数据集。

## util
功能函数模块。
- **metrics**: 定义评估指标。
- **operation**: 包含数据集分割与预处理函数。
- **read_KG**: 读取OWL知识图谱。
- **utils**: 其他功能函数。

## wandb
实验结果记录模块。

## fs_train&generate
处理Few-Shot任务的训练与推理。

## main
项目的主程序脚本。

## zs_generate
处理Zero-Shot推理任务。

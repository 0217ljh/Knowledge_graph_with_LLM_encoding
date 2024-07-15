# 下游任务

## 节点级任务(Node)

* **Task1**: Classification - _光谱上某吸收峰的来源_
  **prompt node**.  node classification on the category of particular absorption peak
  **Input**: Nodalised Spectrum
  **Downstream**: class node.
  * **class node**: 'FG1'/'FG2/FG3...'

  **Output**: probability
  **是否实现**：未实现，需修改如下
  * prompt节点，class节点
  * 任务图构建修改：Node形式
  * 模型不变：GNN+FN

## 边线级任务(Edge)

* **Task2**: Judgment - _光谱上不同吸收峰是否同源_
  **prompt node**.  edge Judgment on the link between twp particular absorption peaks
  **Input**: Nodalised Spectrum
  **Downstream**: class node.
  * **class node**: 'Yes'/'No'

  **Output**: probability
  **是否实现**：未实现，需修改如下
  * prompt节点，class节点
  * 任务图构建修改：Edge形式
  * 模型不变：GNN+FN

## 图级任务(Graph)

* **Task3**: Judgment - _光谱对应的物质是否存在特定官能团_
  **prompt node**.  graph judgement about "Phenyl"(苯基), based on spectral information and knowledge graph: 
  **Input**: Nodalised Spectrum
  **Downstream**: class node
  * **class node**: 'Yes'/'No'

  **Output**: probability
  **是否实现**：未实现，仅需在原框架上修改对应节点内容即可，需修改如下
  * prompt节点，class节点
  * 任务图构建不变
  * 模型不变：GNN+FN

&nbsp;

* **Task4**: Classification -  _光谱对应物质的化学分类_
  **prompt node**. graph classification on the sample's category based on spectral information and knowledge graph
  **Input**: Nodalised Spectrum
  **Downstream**: class node
  * **class node**: 'Aromatic compound'/'Non-aromatic compound'

  **Output**: probability
  **是否实现**：已实现，效果良好

## 复杂下游任务

* **Task5**: Generation - _光谱生成_
  **prompt node**. graph generation based on Functional information and knowledge graph
  **Input**: a serious information of Functional group
  **Downstream**: decoder
  **Output**: 节点形式的光谱？还是光谱图？
  **是否实现**：未实现，需修改对应模型解码形式，需修改如下
  * prompt节点，class节点
  * 任务图格式：待定
  * 模型修改：GNN+decoder
&nbsp;

* **Task6**: Analysis - _分析光谱的可能物质来源与信息_
  **prompt node**. analysis the possible sources of substances and information of spectrum
  **Input**: Nodalised Spectrum
  **Downstream**: text decoder
  **Output**: Text--sources and related information
  **是否实现**：未实现，需修改对应模型解码形式，需修改如下
  * prompt节点，class节点
  * 任务图格式：待定
  * 模型修改：GNN_decoder
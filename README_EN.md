# Project Directory Overview

This document provides a structured overview of the project directories and their functionalities.

---

## Table of Contents
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
Stores prompt graph data for different tasks.
- **FS_Task**: Contains prompt graphs for FS. A prompt graph is formed by integrating knowledge node graphs and spectrum node graphs, adding prompt information, and encoding them into a unified graph structure.

## Configs
Contains configurations for models, tasks, and training.
- **data_config**: Data configuration.
- **default_config**: Default model configuration.
- **task_config**: Task-specific configuration.
- **overRide_task_config**: Overwrites task and model configurations (modifiable).

## Data_process
Data processing and integration module, converting raw data into graph-structured data.
- **data_collect/datatype/knowledge_map**: Processes knowledge graph data.
- **data_collect/datatype/spectrum**: Processes spectrum graph data.
- **Merge_collect, Merge_process**: Integrates graph structures.

## Database
Stores raw data and labels.
- **FS_Database**: FS raw data.
- **Functional group**: Data on functional groups.
- **Knowledge graph**: OWL knowledge graph.
- **Spectrum_database**: Raw spectrum data.
- **Task_set**: Labels for different tasks.

## GNN_LLM_Spectrum
Stores model weights.

## Light
Modules related to training.

## Models
Model-related modules.
- **LLM**: Includes the encoder and model definitions.
- **nn**: Includes GNN and GAT implementations.

## saved_exp
Stores historical experiment parameters.

## Spectrum simulation
Independent module for simulating and generating spectra.

### Submodules
1. **data**: Based on the QM9 molecular dataset, containing physical information for over 100,000 molecules.
2. **LLM**: Contains the model definition and encoder.
3. **code**:
   - **detanet_model**: Includes physical property prediction and spectrum simulation code, as well as training modules.
     - **detanet.py**: Core module for predicting physical properties using the E3 graph neural network.
     - **spectra_simulator**: Simulates spectra based on the physical properties predicted by Detanet.
     - **else**: Training modules.
   - **trained_param**: Stores historical training parameters and model weights.
   - **Dockerfile**: Image file for the module.
   - **main_train**: Main script for training Detanet.
   - **main_calculate**: Main script for predicting physical properties and simulating spectra using trained Detanet.

## task_data
Constructs datasets for sub-tasks.
- **edge**: Constructs datasets for edge-related tasks.
- **graph**: Constructs datasets for graph-related tasks.
  - **cls**: Subgraph classification.
  - **jdm**: Subgraph identification.
- **node**: Constructs datasets for node-related tasks.

## util
Utility functions.
- **metrics**: Defines evaluation metrics.
- **operation**: Contains dataset splitting and preprocessing functions.
- **read_KG**: Reads OWL knowledge graphs.
- **utils**: Other utility functions.

## wandb
Tracks experimental results.

## fs_train&generate
Handles training and inference for few-shot tasks.

## main
Main script for running the project.

## zs_generate
Handles zero-shot inference.

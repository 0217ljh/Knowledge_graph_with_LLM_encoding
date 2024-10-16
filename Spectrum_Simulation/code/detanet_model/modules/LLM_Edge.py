from transformers import BertLMHeadModel, AutoTokenizer
import torch
from torch import nn
from detanet_model.modules import Embedding, Radial_Basis
import re
from torch.nn.utils.rnn import pad_sequence
from .acts import activations
from peft import LoraConfig, get_peft_model

class LLM_Edge_Module(nn.Module):
    def __init__(self, llm_model_name="bert-base-uncased", num_features=40, device='cuda', batch_size=32, num_llm_layers=6, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(LLM_Edge_Module, self).__init__()
        self.num_features = num_features
        self.device = device
        self.batch_size = batch_size  # Set batch size
        self.num_llm_layers = num_llm_layers  # Set number of LLM layers

        # Initialize a pre-trained LLM model and tokenizer (BERT LM Head Model)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, torch_dtype=torch.float32)

        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load LLM model with reduced number of layers
        self.llm = BertLMHeadModel.from_pretrained(llm_model_name, torch_dtype=torch.float32, num_hidden_layers=self.num_llm_layers).to(self.device)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA for fine-tuning
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['query', 'key', 'value'],  # Apply LoRA to attention layers
            lora_dropout=lora_dropout,
            bias='none'
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # Feature embedding layer
        self.feature_embedding = nn.Linear(self.num_features, num_features).to(self.device)

        # Linear layers to project LLM output to desired feature dimensions
        self.linear_proj = nn.Linear(self.llm.config.hidden_size, num_features).to(self.device)
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.linear_proj.weight)  # Xavier uniform initialization
        nn.init.zeros_(self.linear_proj.bias)  # Initialize biases to 0

        # Activation function
        self.eact = nn.ReLU().to(self.device)

    def process_generated_output(self, generated_output, index):
        # Process the generated output and extract edge features
        generated_ids = generated_output

        decoded_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        edge_features = []
        for output in decoded_output:
            feature_values = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            feature_tensor = []

            for value in feature_values:
                try:
                    feature_tensor.append(float(value))
                except ValueError:
                    print(f"Unable to convert to float: {value}")

            if feature_tensor:
                edge_features.append(torch.tensor(feature_tensor, device=self.device))  # Move to device

        if edge_features:
            eij = self.pad_tensors(edge_features)  # Use custom padding function
        else:
            eij = torch.empty(0, device=self.device)

        return eij

    def pad_tensors(self, tensor_list):
        if not tensor_list:
            return torch.empty(0, device=self.device)

        # Use pad_sequence to pad tensors
        padded_tensors = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)

        # Calculate required padding size
        desired_length = 80
        if padded_tensors.size(1) < desired_length:
            padding = torch.zeros(len(tensor_list), desired_length - padded_tensors.size(1), device=padded_tensors.device)
            padded_tensors = torch.cat([padded_tensors, padding], dim=1)

        return padded_tensors

    def format_input_for_llm(self, S, rbf, index):
        """
        Combine node features and edge features into LLM input format (text or feature vector).
        """
        inputs = []
        i, j = index
        for idx_i, idx_j in zip(i, j):
            atom_i_features = S[idx_i].tolist()  # Node i features
            atom_j_features = S[idx_j].tolist()  # Node j features
            edge_features = rbf[idx_i].tolist()  # Use all rbf dimensions as edge features

            # Detailed English prompt
            prompt = (
                "Given the atomic features and the radial basis function (rbf) edge features between two atoms, predict the detailed interaction properties between them.\n"
                "Atom 1 features (type, charge, etc.): {atom_i_features}\n"
                "Atom 2 features (type, charge, etc.): {atom_j_features}\n"
                "Edge (rbf) features: {edge_features}\n"
                "Predict a {num_features}-dimensional feature vector that describes the bond strength, bond order, and orbital interaction between the two atoms."
            ).format(
                atom_i_features=atom_i_features,
                atom_j_features=atom_j_features,
                edge_features=edge_features,
                num_features=self.num_features
            )

            inputs.append(prompt)
        
        return inputs
    
    def forward(self, S, rbf, index):
        """
        Use prompt-based method to compute edge features and process with mini-batch.
        """
        # Format S and rbf into prompt input
        inputs = self.format_input_for_llm(S=S, rbf=rbf, index=index)

        # Mini-batch processing
        eij_list = []
        for batch_start in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[batch_start:batch_start + self.batch_size]
            # Tokenize inputs
            encoded_inputs = self.tokenizer(
                batch_inputs,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded_inputs = {key: val.to(self.device) for key, val in encoded_inputs.items()}

            # Generate edge features using the LLM's generate method
            with torch.no_grad():
                generated_output = self.llm.generate(**encoded_inputs, max_length=512)  # Control generation length

            # Process generated tokens into edge features
            eij = self.process_generated_output(generated_output, index)
            eij_list.append(eij)

        eij = torch.cat(eij_list, dim=0)

        # Split into scalar and tensor components
        mijs2, mijs = torch.split(eij, split_size_or_sections=[self.num_features, self.num_features], dim=-1)

        # Pass edge features through feature embedding layer
        mijs2 = self.eact(self.feature_embedding(mijs2))
        mijs = self.eact(self.feature_embedding(mijs))

        return mijs2, mijs





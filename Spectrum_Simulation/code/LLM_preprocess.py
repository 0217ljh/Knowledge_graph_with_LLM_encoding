from transformers import BertLMHeadModel, AutoTokenizer
import torch
from torch import nn
from detanet_model.modules import Embedding, Radial_Basis
import re
from torch.nn.utils.rnn import pad_sequence

class LLM_Edge_Module(nn.Module):
    def __init__(self, llm_model_name="meta-llama/Llama-2-7b-hf", num_features=40, device='cuda', batch_size=32, num_llm_layers=6):
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
        
        # Linear layers to project LLM output to desired feature dimensions
        self.linear_proj = nn.Linear(self.llm.config.hidden_size, 2 * num_features).to(self.device)
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.linear_proj.weight)  # Xavier uniform initialization
        nn.init.zeros_(self.linear_proj.bias)  # Initialize biases to 0

        # Activation function
        #self.eact = nn.ReLU().to(self.device)

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

        print(f"Padded tensor shape: {padded_tensors.shape}")
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
                "Given the features of two atoms and the edge features, generate the edge features between the two atoms.\n"
                "Features of atom 1: {atom_i_features}\n"
                "Features of atom 2: {atom_j_features}\n"
                "Features of edge: {edge_features}\n"
                "The generated edge features should be a {num_features}-dimensional vector.\n"
                "Generated result:"
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
            
            # # Ensure eij has correct shape before passing to linear layer
            # if eij.shape[1] != self.llm.config.hidden_size:
            #     eij = eij.view(-1, self.llm.config.hidden_size)

            # eij = self.linear_proj(eij)
            # eij = self.eact(eij)
            eij_list.append(eij)

        eij = torch.cat(eij_list, dim=0)

        # Split into scalar and tensor components
        mijs2, mijs = torch.split(eij, split_size_or_sections=[self.num_features, self.num_features], dim=-1)
        
        return mijs2, mijs

    def process_dataset(self, dataset, grad):
        """
        Process entire dataset and extract mijs2 and mijs for each element.
        """
        processed_data = []
        self.Embedding = Embedding(num_features=40, act='swish', device=self.device, max_atomic_number=9).to(self.device)
        self.Radial = Radial_Basis(radial_type='trainable_bessel', num_radial=16, use_cutoff=False).to(self.device)
        
        for data_element in dataset:
            z = data_element.z.to(self.device)
            pos = data_element.pos.to(self.device)
            S = self.Embedding(z)
            i, j = data_element.edge_index

            if grad == 'Hi':
                posa = pos.clone()
                posb = pos.clone()
                posj = posa[j]
                posi = posb[i]
            else:
                posi = pos[i]
                posj = pos[j]

            # From ri-rj we obtain the coordinate difference vector and sum the squares to obtain the interatomic distance.
            rij = posj - posi
            r = torch.norm(rij, dim=-1)
            rbf = self.Radial(r)
            index = data_element.edge_index  # Edge index     
            
            # Use LLM to process each element and extract mijs2 and mijs
            mijs2, mijs = self.forward(S=S, rbf=rbf, index=index)
            
            # Store processed results in an array
            processed_data.append((mijs2, mijs))
        
        return processed_data

def process_data_with_llm_edge(dataset, llm_model_name="bert-base-uncased", device='cuda', grad=None, batch_size=32, num_llm_layers=6):
    # Initialize LLM_Edge_Module
    llm_edge_module = LLM_Edge_Module(llm_model_name=llm_model_name, device=device, batch_size=batch_size, num_llm_layers=num_llm_layers)
    
    # Process dataset and extract mijs2 and mijs
    processed_data = llm_edge_module.process_dataset(dataset, grad)
    
    # Return processed data (mijs2, mijs arrays)
    return processed_data






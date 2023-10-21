from transformers import (
    BertPreTrainedModel, 
    BertModel, 
    RobertaPreTrainedModel,
    RobertaModel,
) 
from transformers.modeling_outputs import TokenClassifierOutput
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss


class BertNegSampleForTokenClassification(BertPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, mlp_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config, add_pooling_layer=False)
        self.pooling = nn.Sequential(
            nn.Linear(config.hidden_size * 4, mlp_config["hidden_size"]),
            nn.Tanh(),            
        )
        self.cls = nn.Sequential(
            nn.Dropout(mlp_config["dropout_rate"]),
            nn.Linear(mlp_config["hidden_size"], self.num_labels),
        )
        self.post_init()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)
        logits = self.cls(hidden_states)
        
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


class RobertaNegSampleForTokenClassification(RobertaPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, mlp_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.pooling = nn.Sequential(
            nn.Linear(config.hidden_size * 4, mlp_config["hidden_size"]),
            nn.Tanh(),            
        )
        self.cls = nn.Sequential(
            nn.Dropout(mlp_config["dropout_rate"]),
            nn.Linear(mlp_config["hidden_size"], self.num_labels),
        )
        self.post_init()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)
        logits = self.cls(hidden_states)
        
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
        
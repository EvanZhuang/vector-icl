import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel, PretrainedConfig


class ProjectorConfig(PretrainedConfig):
    def __init__(self, in_features=4096, out_features=None, expansion_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.expansion_ratio = expansion_ratio


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MLPProjector(PreTrainedModel):
    def __init__(self, config: ProjectorConfig, dtype=torch.float32):
        super().__init__(config)
        in_features, out_features, expansion_ratio = config.in_features, config.out_features, config.expansion_ratio
        self.config = config
        if out_features is None:
            out_features = in_features
        mlp_bias = False
        self.input_proj = nn.Linear(in_features, out_features, bias=mlp_bias)
        self.layer_norm = LlamaRMSNorm(in_features)
        self.gate_proj = nn.Linear(out_features, out_features*expansion_ratio, bias=mlp_bias)
        self.up_proj = nn.Linear(out_features, out_features*expansion_ratio, bias=mlp_bias)
        self.down_proj = nn.Linear(out_features*expansion_ratio, out_features, bias=mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, input_x):
        x = self.layer_norm(input_x)
        x = self.input_proj(x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj + x


class LinearProjector(PreTrainedModel):
    def __init__(self, config: ProjectorConfig, dtype=torch.float32, **kwargs):
        super().__init__(config)
        in_features, out_features = config.in_features, config.out_features
        self.config = config
        if out_features is None:
            out_features = in_features
        self.linear = nn.Linear(in_features, out_features, dtype=dtype, bias=False)

    def forward(self, x):
        return self.linear(x)


class ModelWrapper(PreTrainedModel):
    def __init__(self, model: PreTrainedModel, config: PretrainedConfig = None, max_length=512, tokenizer=None):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, text_lst, **kwargs):
        input_ids = self.tokenizer(text_lst, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)['input_ids'].to(self.model.device)
        return self._encode(input_ids, **kwargs)
    
    def _encode(self, input_ids, attention_mask=None, **kwargs):
        # return the last hidden states
        return self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].mean(dim=1, keepdim=True)



class EncoderWrapper:
    def __init__(self, model: None, model_name_or_path=None, max_length=512, batch_size=8):
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, text_lst, max_batch_size=None):
        if max_batch_size is None:
            max_batch_size = self.batch_size
        main_text_embeds = []
        for enc_batch in range(len(text_lst)//max_batch_size + 1):
            start_idx = enc_batch * max_batch_size
            end_idx = min((enc_batch + 1) * max_batch_size, len(text_lst))
            if start_idx == end_idx:
                break
            with torch.no_grad():
                if self.model_name_or_path == "nvidia/NV-Embed-v1":
                    encode_states = self.model.encode(text_lst[start_idx:end_idx], instruction="", max_length=self.max_length).unsqueeze(1).detach()
                elif self.model_name_or_path == "sentence-transformers/gtr-t5-base":
                    encode_states = self.model.encode(text_lst[start_idx:end_idx], device=self.model.device, convert_to_tensor=True).unsqueeze(1).detach()
                elif self.model_name_or_path == "dunzhang/stella_en_1.5B_v5":
                    encode_states = self.model.encode(text_lst[start_idx:end_idx], device=self.model.device, convert_to_tensor=True).unsqueeze(1).detach()
                elif self.model_name_or_path == "Salesforce/SFR-Embedding-2_R":
                    encode_states = self.model.encode(text_lst[start_idx:end_idx], device=self.model.device, convert_to_tensor=True).unsqueeze(1).detach()
            main_text_embeds.append(encode_states)
        last_states = torch.cat(main_text_embeds, dim=0)
        return last_states
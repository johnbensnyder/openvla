"""
openvla_nxd_pp.py

OpenVLA model with Pipeline Parallelism (PP) + Tensor Parallelism (TP) for Trainium.
Splits model into 2 pipeline stages to reduce per-device memory.

Stage 0: Vision backbone + Projector + Embedding + first N/2 LLM layers
Stage 1: Last N/2 LLM layers + LM head
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoConfig

import neuronx_distributed as nxd
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.lora import LoraConfig as NxDLoraConfig


def get_nxd_lora_config(lora_rank: int = 32, lora_alpha: int = 16, lora_dropout: float = 0.0):
    """Create NxD LoRA configuration."""
    return NxDLoraConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        save_lora_base=False,
        merge_lora=False,
    )


class VisionEmbeddingStage(nn.Module):
    """Pipeline Stage 0: Vision + Projector + Embedding + First half of LLM layers."""
    
    def __init__(self, config, vla_path: str, num_layers: int):
        super().__init__()
        from prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone, PrismaticProjector
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
        
        self.config = config
        self.num_layers = num_layers
        
        # Vision backbone
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )
        
        # Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )
        
        # LLM embedding
        self.embed_tokens = nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
        )
        
        # First half of decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config.text_config, layer_idx=i)
            for i in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Vision encoding
        patch_features = self.vision_backbone(pixel_values)
        projected_patches = self.projector(patch_features)
        
        # Text embedding
        input_embeds = self.embed_tokens(input_ids)
        
        # Combine: [BOS, patches, text]
        hidden_states = torch.cat([
            input_embeds[:, :1, :],
            projected_patches,
            input_embeds[:, 1:, :]
        ], dim=1)
        
        # Build attention mask
        batch_size, seq_len = hidden_states.shape[:2]
        if attention_mask is not None:
            patch_mask = torch.ones(batch_size, projected_patches.shape[1], 
                                   dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1)
        
        # Process through first half of layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        # Pass hidden states and metadata to next stage
        return hidden_states, attention_mask, labels


class LMHeadStage(nn.Module):
    """Pipeline Stage 1: Second half of LLM layers + LM head."""
    
    def __init__(self, config, start_layer: int, num_layers: int):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
        
        self.config = config
        
        # Second half of decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config.text_config, layer_idx=start_layer + i)
            for i in range(num_layers)
        ])
        
        # Final norm and LM head
        self.norm = LlamaRMSNorm(config.text_config.hidden_size)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
    def forward(self, hidden_states, attention_mask=None, labels=None):
        # Process through second half of layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, logits


class OpenVLAPipelineModel(nn.Module):
    """OpenVLA with Pipeline Parallelism using NxD."""
    
    def __init__(
        self,
        vla_path: str,
        tensor_parallel_size: int = 4,
        pipeline_parallel_size: int = 2,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        num_microbatches: int = 1,
    ):
        super().__init__()
        
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        self.num_microbatches = num_microbatches
        
        # Load config
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        config = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
        self.config = config
        
        # Llama-7B has 32 layers, split evenly
        total_layers = config.text_config.num_hidden_layers
        layers_per_stage = total_layers // pipeline_parallel_size
        
        # Get pipeline rank
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        
        # Create stage based on PP rank
        if pp_rank == 0:
            self.stage = VisionEmbeddingStage(config, vla_path, layers_per_stage)
        else:
            start_layer = pp_rank * layers_per_stage
            self.stage = LMHeadStage(config, start_layer, layers_per_stage)
        
        self.stage = self.stage.to(torch.bfloat16)
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """Forward pass - NxD PP handles inter-stage communication."""
        return self.stage(input_ids, attention_mask, pixel_values, labels)


def load_openvla_pipeline(
    vla_path: str,
    tensor_parallel_size: int = 4,
    pipeline_parallel_size: int = 2,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """
    Load OpenVLA with Pipeline + Tensor Parallelism.
    
    With TP=4, PP=2 on trn1.32xlarge (8 workers):
    - Workers 0-3: Stage 0 (Vision + first 16 LLM layers), TP across 4 devices
    - Workers 4-7: Stage 1 (last 16 LLM layers + head), TP across 4 devices
    """
    # Initialize parallel state
    nxd.parallel_layers.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
    )
    
    lora_config = get_nxd_lora_config(lora_rank, lora_alpha, lora_dropout)
    
    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        lora_config=lora_config,
        pipeline_config={
            "num_microbatches": 1,
            "output_loss_value_spec": True,
            "return_mb_loss": True,
        },
    )
    
    model = nxd.initialize_parallel_model(
        nxd_config,
        model_fn=lambda: OpenVLAPipelineModel(
            vla_path=vla_path,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ),
    )
    
    return model

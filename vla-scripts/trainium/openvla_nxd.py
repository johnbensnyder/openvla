"""
openvla_nxd.py

OpenVLA model wrapper for NeuronX Distributed (NxD) tensor parallelism on Trainium.
Applies tensor parallelism to the LLM backbone and NxD LoRA to linear layers.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import AutoConfig, AutoModelForCausalLM

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import layers as nxd_layers
from neuronx_distributed.modules.lora import LoraConfig as NxDLoraConfig


def get_nxd_lora_config(lora_rank: int = 32, lora_alpha: int = 16, lora_dropout: float = 0.0):
    """Create NxD LoRA configuration for OpenVLA LLM backbone."""
    return NxDLoraConfig(
        enable_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        lora_verbose=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        save_lora_base=False,
        merge_lora=False,
    )


class OpenVLAForTrainium(nn.Module):
    """
    OpenVLA model adapted for Trainium with tensor parallelism.
    
    Architecture:
    - Vision backbone: Runs on single device (not parallelized)
    - Projector: Runs on single device
    - LLM backbone: Tensor-parallelized across NeuronCores with NxD LoRA
    """
    
    def __init__(
        self,
        vla_path: str,
        tensor_parallel_size: int = 8,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.tensor_parallel_size = tensor_parallel_size
        
        # Load config to get model structure
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import (
            PrismaticVisionBackbone, PrismaticProjector
        )
        
        config = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
        self.config = config
        
        # Vision backbone - not parallelized (relatively small)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )
        
        # Projector - not parallelized
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )
        
        # LLM backbone with NxD tensor parallelism and LoRA
        lora_config = get_nxd_lora_config(lora_rank, lora_alpha, lora_dropout)
        
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=tensor_parallel_size,
            lora_config=lora_config,
        )
        
        # Initialize LLM with tensor parallelism
        self.language_model = nxd.initialize_parallel_model(
            nxd_config,
            model_fn=lambda: AutoModelForCausalLM.from_pretrained(
                vla_path,
                subfolder="language_model" if hasattr(config, 'text_config') else None,
                config=config.text_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ),
        )
        
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        
        # For action un-normalization during inference
        self.norm_stats = None
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def freeze_non_lora_params(self):
        """Freeze all parameters except LoRA adapters."""
        # Vision backbone - freeze entirely
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # Projector - freeze entirely  
        for param in self.projector.parameters():
            param.requires_grad = False
            
        # LLM - NxD LoRA handles freezing automatically
        # Only LoRA params will have requires_grad=True
        
    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ):
        """Forward pass for training."""
        # Visual feature extraction (not parallelized)
        patch_features = self.vision_backbone(pixel_values)
        projected_patch_embeddings = self.projector(patch_features)
        
        # Get input embeddings from LLM
        input_embeddings = self.get_input_embeddings()(input_ids)
        
        # Build multimodal embeddings: [BOS, image_patches, text_tokens]
        multimodal_embeddings = torch.cat([
            input_embeddings[:, :1, :],
            projected_patch_embeddings,
            input_embeddings[:, 1:, :]
        ], dim=1)
        
        # Build attention mask for multimodal input
        if attention_mask is not None:
            patch_attention_mask = torch.ones(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            multimodal_attention_mask = torch.cat([
                attention_mask[:, :1],
                patch_attention_mask,
                attention_mask[:, 1:]
            ], dim=1)
        else:
            multimodal_attention_mask = None
            
        # Build labels with IGNORE_INDEX for image patches
        IGNORE_INDEX = -100
        if labels is not None:
            patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat([
                labels[:, :1],
                patch_labels,
                labels[:, 1:]
            ], dim=1)
        else:
            multimodal_labels = None
            
        # Forward through LLM (tensor-parallelized)
        output = self.language_model(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_attention_mask,
            labels=multimodal_labels,
            use_cache=False,
        )
        
        return output


def load_openvla_for_trainium(
    vla_path: str,
    tensor_parallel_size: int = 8,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> OpenVLAForTrainium:
    """
    Load OpenVLA model configured for Trainium training.
    
    Args:
        vla_path: HuggingFace model path (e.g., "openvla/openvla-7b")
        tensor_parallel_size: Number of tensor parallel ranks (default: 8)
        lora_rank: LoRA rank (default: 32)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: LoRA dropout (default: 0.0)
        
    Returns:
        OpenVLAForTrainium model with LoRA enabled
    """
    model = OpenVLAForTrainium(
        vla_path=vla_path,
        tensor_parallel_size=tensor_parallel_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.freeze_non_lora_params()
    return model

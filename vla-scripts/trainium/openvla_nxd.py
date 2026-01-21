"""
openvla_nxd.py

MiniVLA/OpenVLA model wrapper for NeuronX Distributed (NxD) on Trainium.
Supports both OpenVLA (Llama-7B) and MiniVLA (Qwen2.5-0.5B) backbones.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoConfig, AutoModelForCausalLM

import neuronx_distributed as nxd
from neuronx_distributed.modules.lora import LoraConfig as NxDLoraConfig


def get_nxd_lora_config(lora_rank: int = 32, lora_alpha: int = 16, lora_dropout: float = 0.0):
    """Create NxD LoRA configuration for VLA LLM backbone."""
    return NxDLoraConfig(
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
    OpenVLA/MiniVLA model adapted for Trainium.
    
    Architecture:
    - Vision backbone: Runs on CPU (not parallelized)
    - Projector: Runs on CPU
    - LLM backbone: On device with NxD LoRA
    
    Supports both OpenVLA (Llama-7B) and MiniVLA (Qwen2.5-0.5B).
    """
    
    def __init__(
        self,
        vla_path: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        num_microbatches: int = 1,
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
        
        # Vision backbone - runs on CPU
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )
        
        # Projector - runs on CPU
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )
        
        # Detect LLM type from config
        llm_type = config.text_config.model_type
        print(f"Detected LLM backbone: {llm_type}")
        
        # LLM backbone with NxD LoRA
        lora_config = get_nxd_lora_config(lora_rank, lora_alpha, lora_dropout)
        
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=1,  # MiniVLA fits on single core
            pipeline_parallel_size=1,
            lora_config=lora_config,
        )
        
        # Initialize LLM based on backbone type
        if llm_type == "qwen2":
            from transformers import Qwen2ForCausalLM
            self.language_model = nxd.initialize_parallel_model(
                nxd_config,
                model_fn=lambda: Qwen2ForCausalLM(config.text_config).to(torch.bfloat16),
            )
        else:  # Default to Llama for OpenVLA
            from transformers import LlamaForCausalLM
            self.language_model = nxd.initialize_parallel_model(
                nxd_config,
                model_fn=lambda: LlamaForCausalLM(config.text_config).to(torch.bfloat16),
            )
        
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.pipeline_parallel_size = pipeline_parallel_size
        self.llm_type = llm_type
        
        # For action un-normalization during inference
        self.norm_stats = None
        
    def load_pretrained_weights(self, vla_path: str):
        """Load pretrained weights from VLA checkpoint."""
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import json
        
        # Try sharded format first, fall back to single file
        try:
            index_file = hf_hub_download(vla_path, "model.safetensors.index.json")
            with open(index_file) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            state_dict = {}
            for shard in shard_files:
                shard_path = hf_hub_download(vla_path, shard)
                state_dict.update(load_file(shard_path))
        except Exception:
            # Single file format
            model_file = hf_hub_download(vla_path, "model.safetensors")
            state_dict = load_file(model_file)
        
        # Load vision backbone weights
        vision_state = {k.replace("vision_backbone.", ""): v for k, v in state_dict.items() 
                       if k.startswith("vision_backbone.")}
        self.vision_backbone.load_state_dict(vision_state, strict=False)
        
        # Load projector weights
        proj_state = {k.replace("projector.", ""): v for k, v in state_dict.items()
                     if k.startswith("projector.")}
        self.projector.load_state_dict(proj_state, strict=False)
        
        # Load LLM weights
        llm_state = {k.replace("language_model.", ""): v for k, v in state_dict.items()
                    if k.startswith("language_model.")}
        self.language_model.load_state_dict(llm_state, strict=False)
        
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
        projected_patch_embeddings: Optional[torch.Tensor] = None,
    ):
        """Forward pass for training.
        
        If projected_patch_embeddings is provided, skip vision encoding (for CPU offload).
        """
        if projected_patch_embeddings is None:
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
    pipeline_parallel_size: int = 1,
    num_microbatches: int = 1,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> OpenVLAForTrainium:
    """
    Load VLA model configured for Trainium training.
    """
    model = OpenVLAForTrainium(
        vla_path=vla_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        num_microbatches=num_microbatches,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.load_pretrained_weights(vla_path)
    model.freeze_non_lora_params()
    return model

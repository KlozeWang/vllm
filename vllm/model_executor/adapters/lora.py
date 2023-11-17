from peft.tuners.lora import LoraLayer
from peft import PeftConfig, LoraConfig, PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils import WEIGHTS_NAME
from huggingface_hub import hf_hub_download
import torch
from torch import nn
from vllm.model_executor.models.opt import ColumnParallelLinear,RowParallelLinear
from .mapping import MODEL_LAYER_MAPPING
from operator import attrgetter
import os

class VllmLoRA_MLP_DOWN(LoraLayer, RowParallelLinear):
    def __init__(
            self,
            input_size,
            output_size,
            *args,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            adapter_name="default",
            **kwargs,
        ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        RowParallelLinear.__init__(self, input_size, output_size, *args, **kwargs)
        LoraLayer.__init__(self, input_size, output_size)

        nn.Linear.reset_parameters(self)
        
        self.update_layer("down_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("up_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("down_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("gate_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, input_):
        result, _ = RowParallelLinear.forward(self, input_)
        x = input_.to(self.lora_A["down_proj"].weight.dtype)
        adapter = "down_proj"
        lora_results = self.lora_B[adapter](self.lora_A[adapter](self.lora_dropout[adapter](x))) * self.scaling[adapter]
        result += lora_results
        return result, _
    
class VLlmLoRA_MLP_GATE_UP(LoraLayer, ColumnParallelLinear):
    def __init__(
            self,
            input_size,
            output_size,
            *args,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            adapter_name="default",
            **kwargs,
        ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        ColumnParallelLinear.__init__(self, input_size, output_size, *args, **kwargs)
        LoraLayer.__init__(self, input_size, output_size//2)

        nn.Linear.reset_parameters(self)
        
        self.update_layer("gate_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        self.update_layer("up_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("up_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("down_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("gate_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, input_):
        result, _ = ColumnParallelLinear.forward(self, input_)
        x = input_.to(self.lora_A["gate_proj"].weight.dtype)
        lora_results = [
            self.lora_B[adapter](self.lora_A[adapter](self.lora_dropout[adapter](x)))
            * self.scaling[adapter]
            for adapter in ["gate_proj","up_proj"]
        ]
        result += torch.cat(lora_results, dim=-1)
        return result, _

class VllmLoRA_QKV(LoraLayer, ColumnParallelLinear):
    def __init__(
        self,
        input_size,
        output_size,
        *args,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        adapter_name="default",
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        ColumnParallelLinear.__init__(self, input_size, output_size, *args, **kwargs)
        LoraLayer.__init__(self, input_size, input_size)

        nn.Linear.reset_parameters(self)
        self.update_layer("q_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        self.update_layer("k_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        self.update_layer("v_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("up_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("down_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        # self.update_layer("gate_proj", r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, input_):
        result, _ = ColumnParallelLinear.forward(self, input_)
        x = input_.to(self.lora_A["q_proj"].weight.dtype)
        lora_results = [
            self.lora_B[adapter](self.lora_A[adapter](self.lora_dropout[adapter](x)))
            * self.scaling[adapter]
            for adapter in ["q_proj","k_proj","v_proj"]# ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
        ]
        result += torch.cat(lora_results, dim=-1)
        return result, _



class LoRAModel:
    @classmethod
    def from_pretrained(cls, model, adapter_model_name: str, **kwargs):
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig.from_pretrained(
                adapter_model_name, subfolder=kwargs.get("subfolder", None)
            ).peft_type
        ].from_pretrained(adapter_model_name, subfolder=kwargs.get("subfolder", None))

        cls_name = attrgetter("__class__.__name__")(model)
        layers = attrgetter(MODEL_LAYER_MAPPING[cls_name])(model)

        cls.load_adapter(layers, config)
        cls._update_weights(model, adapter_model_name, config, layers)

    @classmethod
    def load_adapter(cls, layers, config: LoraConfig):
        for layer in layers:
            qkv_proj = layer.self_attn.qkv_proj
            new_model = VllmLoRA_QKV(
                input_size=qkv_proj.input_size,
                output_size=qkv_proj.output_size,
                r=config.r,
                lora_alpha=config.lora_alpha,
            )
            cls._replace_module(
                layer.self_attn, "qkv_proj", new_model, layer.self_attn.qkv_proj
            )

            gate_up_proj = layer.mlp.gate_up_proj
            new_model = VLlmLoRA_MLP_GATE_UP(
                input_size=gate_up_proj.input_size,
                output_size=gate_up_proj.output_size,
                r=config.r,
                lora_alpha=config.lora_alpha,
            )
            cls._replace_module(
                layer.mlp, "gate_up_proj", new_model, layer.mlp.gate_up_proj
            )
    
            down_proj = layer.mlp.down_proj
            
            new_model = VllmLoRA_MLP_DOWN(
                input_size=down_proj.input_size,
                output_size=down_proj.output_size,
                r=config.r,
                lora_alpha=config.lora_alpha,
            )
            cls._replace_module(
                layer.mlp, "down_proj", new_model, layer.mlp.down_proj
            )

    @classmethod
    def _replace_module(cls, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    @classmethod
    def _update_weights(cls, model, adapter_model_name, config, layers):
        if os.path.exists(os.path.join(adapter_model_name, WEIGHTS_NAME)):
            filename = os.path.join(adapter_model_name, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(adapter_model_name, WEIGHTS_NAME)
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {adapter_model_name} in {adapter_model_name} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {adapter_model_name}."
                )

        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        peft_model_state_dict = {}
        # base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight
        #    ↓ ↓ ↓
        # model.decoder.layers.0.self_attn.qkv_proj.lora_A.q_proj.weight
        peft_prefix = list(adapters_weights.keys())[0].split("layers")[0]
        vllm_prefix = ""
        for k, _ in model.named_parameters():
            if "lora_" in k:
                vllm_prefix = k.split("layers")[0]
        if not vllm_prefix:
            raise ValueError("Can't find vllm_prefix.")

        for i in range(len(layers)):
            for lora in ["lora_A", "lora_B"]:
                for target in config.target_modules:
                    if target in ["q_proj","k_proj","v_proj"]:
                        proj_key = (
                            f"{peft_prefix}layers.{i}.self_attn.{target}.{lora}.weight"
                        )
                        proj_weight = adapters_weights.get(proj_key)

                        to_key = f"{vllm_prefix}layers.{i}.self_attn.qkv_proj.{lora}.{target}.weight"
                        peft_model_state_dict[to_key] = proj_weight
                    elif target in ["gate_proj","up_proj","down_proj"]: 
                        proj_key = (
                            f"{peft_prefix}layers.{i}.mlp.{target}.{lora}.weight"
                        )
                        proj_weight = adapters_weights.get(proj_key)

                        layer_name = "gate_up_proj"
                        if target == "down_proj":
                            layer_name = "down_proj"
                        to_key = f"{vllm_prefix}layers.{i}.mlp.{layer_name}.{lora}.{target}.weight"
                        peft_model_state_dict[to_key] = proj_weight
        incompatible_keys = model.load_state_dict(peft_model_state_dict, strict=False)
        return incompatible_keys
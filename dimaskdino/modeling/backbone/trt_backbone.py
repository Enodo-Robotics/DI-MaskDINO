import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt as trt
import tensorrt_libs  # noqa: F401 — loads TRT shared libraries

from detectron2.layers import ShapeSpec

logger = logging.getLogger(__name__)

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_INPUT_NAME = "input"
_INPUT_SHAPE = (1, 3, 800, 1216)
_OUTPUT_NAMES = ["res2", "res3", "res4", "res5"]
_OUTPUT_CHANNELS = {"res2": 192, "res3": 384, "res4": 768, "res5": 1536}
_OUTPUT_STRIDES = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}


class TrtBackbone(nn.Module):
    """
    Drop-in replacement for D2SwinTransformer that runs the Swin backbone
    through a pre-built TensorRT FP16 engine.

    The engine has a fixed input shape (1, 3, 800, 1216). Inputs of other
    spatial sizes are bilinearly resized before execution. Outputs are the
    same four feature-map dict that D2SwinTransformer produces.
    """

    def __init__(self, engine_path: str, out_features: list[str]) -> None:
        super().__init__()
        self._out_features = out_features

        runtime = trt.Runtime(_TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        self._output_buffers: dict[str, torch.Tensor] = {
            name: torch.zeros(
                tuple(self._engine.get_tensor_shape(name)),
                dtype=torch.float32,
                device="cuda",
            )
            for name in _OUTPUT_NAMES
        }

        logger.info("TrtBackbone loaded engine from %s", engine_path)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        target_h, target_w = _INPUT_SHAPE[2], _INPUT_SHAPE[3]
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        x = x.contiguous().to(dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream

        self._context.set_tensor_address(_INPUT_NAME, x.data_ptr())
        for name in _OUTPUT_NAMES:
            self._context.set_tensor_address(name, self._output_buffers[name].data_ptr())

        self._context.execute_async_v3(stream)

        return {
            name: self._output_buffers[name].clone()
            for name in self._out_features
            if name in self._output_buffers
        }

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=_OUTPUT_CHANNELS[name],
                stride=_OUTPUT_STRIDES[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return 32

# Optional low-memory Conv3d inference adapter

`resectionlab.low_memory_conv3d` provides CPU/FP32 output-depth tiling for `Conv3d`. It preserves each output cell's full input receptive field and assembles the complete feature map before downstream operations such as InstanceNorm. The module imports without PyTorch; calling its functions requires PyTorch. It is an inference-only numerical tool, not a patient segmentation model or clinical validator.

`conv3d_tiled_depth` uses the original weight, bias, stride, padding, dilation and groups. `attach_tiled_conv3d` patches each unique `torch.nn.Conv3d.forward` in an already loaded, evaluation-mode network. It leaves module objects, parameters, state-dict storage aliases and other operations unchanged. Both functions reject unsupported configurations rather than approximating them: only CPU FP32, explicit zero padding and depth padding no larger than `dilation_depth × (kernel_depth−1)` are admitted; the wrapper also rejects custom or previously patched convolutions. Attach only after loading verified weights and before inference.

```python
import torch
from resectionlab.low_memory_conv3d import attach_tiled_conv3d

network.eval()  # CPU, FP32, with weights already loaded and checked
receipt = attach_tiled_conv3d(network, output_depth_tile=4)
with torch.inference_mode():
    logits = network(generated_input)
```

The implementation was promoted byte-for-byte from the reviewed candidate. Its six algorithm-function ASTs match the earlier generated-control operator/wrapper after accounting for lazy Torch imports and two fail-closed depth-padding guards. Focused generated tests pass in the project Torch environment; without Torch, test collection succeeds and the import-only check passes while tensor tests skip.

One independently audited, generated `[1,2,32,32,64]` nnU-Net control produced 196,608 byte-identical native and tiled logits with unchanged checkpoint and input. Sampled peak process-group RSS was 1,212,137,472 B native and 670,138,368 B tiled in separate one-shot runs. These measurements are specific to that input and host state. The earlier, larger `[1,2,32,64,64]` native control hit warning-level macOS memory pressure and remains a negative result. Neither run establishes 128³ patch feasibility, patient segmentation accuracy, or clinical benefit. The tested generated pattern also had little logit mass near the zero threshold, so threshold-sensitive behavior still needs separate evaluation.

The compact [integration record](../artifacts/low-memory-conv3d-integration-v1/lineage.json) binds source and review hashes; patient data, weights and generated outputs remain outside Git.

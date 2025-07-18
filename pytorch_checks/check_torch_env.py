import torch
print("------ CUDA -------")
print("torch.backends.cuda.is_built()                                   - ", torch.backends.cuda.is_built())
print("torch.backends.cuda.matmul.allow_tf3                             - ", torch.backends.cuda.matmul.allow_tf32)
print("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reductio - ", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
print("torch.backends.cuda.flash_sdp_enabled(                           - ", torch.backends.cuda.flash_sdp_enabled())
print("torch.backends.cuda.mem_efficient_sdp_enabled(                   - ", torch.backends.cuda.mem_efficient_sdp_enabled())
print("torch.backends.cuda.math_sdp_enabled()                           - ", torch.backends.cuda.math_sdp_enabled())
print("------ CUDNN ------")
print("torch.backends.cudnn.version()       - ", torch.backends.cudnn.version())
print("torch.backends.cudnn.is_available()  - ", torch.backends.cudnn.is_available())
print("torch.backends.cudnn.enabled         - ", torch.backends.cudnn.enabled)
print("torch.backends.cudnn.allow_tf32      - ", torch.backends.cudnn.allow_tf32)
print("torch.backends.cudnn.deterministic   - ", torch.backends.cudnn.deterministic)
print("torch.backends.cudnn.benchmark       - ", torch.backends.cudnn.benchmark)
print("torch.backends.cudnn.benchmark_limit - ", torch.backends.cudnn.benchmark_limit)
#print(" - ", )








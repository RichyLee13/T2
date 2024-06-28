# 导入PyTorch库，用于神经网络的构建、优化和量化
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant

# 从指定模块导入轻量级网络模型
from model.net import LightWeightNetwork

# 初始化轻量级网络模型
model = LightWeightNetwork()

# 加载预先训练好的模型权重
model.load_state_dict(torch.load('result_WS/0_ICPR_Track2_UNet_26_06_2024_15_27_42_wDS/model_weight.pth.tar')['state_dict'])

# 设置模型的量化配置，这里选择使用fbgemm作为量化器
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 对模型进行动态量化，将浮点模型转换为量化模型
quantized_model = quant.quantize_dynamic(model)

# 保存量化后的模型权重
torch.save(quantized_model.state_dict(), 'quantized_model.pth')

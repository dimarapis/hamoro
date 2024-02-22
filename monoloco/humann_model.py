import torch.nn as nn
import torch
import math
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
import numpy as np

class TransformerHumaNN(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        #self.w1_modified = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(TransformerLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        #self.w2_modified = nn.Linear(self.linear_size, self.output_size)
        #self.w2_2_modified = nn.Linear(self.linear_size * 2, self.output_size)
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        #self.pre_w2 = nn.Linear(17, self.linear_size)
        
        #self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        #self.batch_norm2 = nn.BatchNorm1d(self.linear_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)
        #self.relu2 = nn.ReLU(inplace=True)
        #self.dropout2 = nn.Dropout(self.p_dropout)
        #self.input_size = input_size
        
    def forward(self, x):
        # pre-processing
        #print(x.dtype)
        x_color = x[:,:34]
        #x_depth = x[:,34:]
        #print(torch.min(x_depth),torch.max(x_depth))
        
        #print(self.input_size, x.shape)
        #print(x.shape)
        
        y = self.w1(x_color)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        # linear layers
        #print(y.shape)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        #print(y.shape)
        y = self.w2(y)


        #pred_y = self.pre_w2(x_depth)
        #pred_y = self.batch_norm2(pred_y)
        #pred_y = self.relu2(pred_y)
        #pred_y = self.dropout2(pred_y)
        
        
        #print('y',torch.min(y),torch.max(y))
        #print('pred_y',torch.min(pred_y),torch.max(pred_y))
        #y = self.w2(y)
        #y = y + pred_y
        #y = torch.cat((y, pred_y), dim=1)

        #y = self.w2_2_modified(y)


        return y



###Contribution : Added TransformerModel class and PositionalEncoding class
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()

        # Positional encoding
        self.positional_encoding = PositionalEncoding(input_size)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=input_size, nhead=num_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout
        )

    def forward(self, x,prev):
        x = self.positional_encoding(x)
        prev = self.positional_encoding(prev)
        x = x.permute(1, 0, 2)
        prev = prev.permute(1, 0, 2)
        output = self.transformer(x, prev)
        output = output.permute(1, 0, 2)
        output = output.squeeze()
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_length=1594):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * (-math.log(10000.0) / input_size))
        pe = torch.zeros(max_length, input_size)
        pe[:, 0::2] = torch.sin(position * div_term[:input_size // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:input_size // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_length, input_size = x.size()
        pe = self.pe[:, :seq_length, :input_size]
        x = x + pe
        return x

class MyLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        

        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if y.shape != x.shape:
            y.unsqueeze_(0)
        #print(f'shape{y.shape}, x_shape{x.shape}')
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
    
    
class TransformerLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        #self.w2 = nn.Linear(self.l_size, self.l_size)
        
        self.w2 = TransformerModel(self.l_size, self.l_size, num_layers=1, num_heads=1, dropout=p_dropout)

        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(x,y)
        if y.shape != x.shape:
            y.unsqueeze_(0)
        #print(f'shape{y.shape}, x_shape{x.shape}')
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
class ModifiedHumanModel_2(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(51, self.linear_size)
        #self.w1_modified = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        #self.w2_modified = nn.Linear(self.linear_size, self.output_size)
        #self.w2_2_modified = nn.Linear(self.linear_size * 2, self.output_size)
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        #self.pre_w2 = nn.Linear(17, self.linear_size)
        
        #self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        #self.batch_norm2 = nn.BatchNorm1d(self.linear_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)
        #self.relu2 = nn.ReLU(inplace=True)
        #self.dropout2 = nn.Dropout(self.p_dropout)
        #self.input_size = input_size
        
    def forward(self, x):
        # pre-processing
        #print(x.dtype)
        #x_color = x[:,:34]
        #x_depth = x[:,34:]
        #print(torch.min(x_depth),torch.max(x_depth))
        
        #print(self.input_size, x.shape)
        #print(x.shape)
        
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        # linear layers
        #print(y.shape)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        #print(y.shape)
        y = self.w2(y)


        #pred_y = self.pre_w2(x_depth)
        #pred_y = self.batch_norm2(pred_y)
        #pred_y = self.relu2(pred_y)
        #pred_y = self.dropout2(pred_y)
        
        
        #print('y',torch.min(y),torch.max(y))
        #print('pred_y',torch.min(pred_y),torch.max(pred_y))
        #y = self.w2(y)
        #y = y + pred_y
        #y = torch.cat((y, pred_y), dim=1)

        #y = self.w2_2_modified(y)


        return y
    
    

class TestModel_3s(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ModifiedLinearBlock(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)

    def forward(self, x):
        x_color = x[:,:34]
        #x_depth = x[:,34:]   
        y = self.w1(x_color)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)

        return y



class TestModel_3s_d(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(51, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ModifiedLinearBlock(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)

    def forward(self, x):
        #x_color = x[:,:34]
        #x_depth = x[:,34:]   
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)

        return y
class TestModel_3m(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.w1m = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.batch_norm1m = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ModifiedLinearBlock(2*self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(2*self.linear_stages)

        self.w2 = nn.Linear(2*self.linear_size, self.output_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)

        self.relu1m = nn.ReLU(inplace=True)
        self.dropout1m = nn.Dropout(self.p_dropout)

    def forward(self, x):
        x_color = x[:,:34]
        y = self.w1(x_color)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        
        x_colorm = x[:,:34] 
        ym = self.w1m(x_colorm)
        ym = self.batch_norm1m(ym)
        ym = self.relu1m(ym)
        ym = self.dropout1m(ym)

        y = torch.cat((y, ym), dim=1)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)

        return y

class TestModel_3d(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.w1m = nn.Linear(51, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.batch_norm1m = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ModifiedLinearBlock(2*self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(2*self.linear_stages)

        self.w2 = nn.Linear(2*self.linear_size, self.output_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)

        self.relu1m = nn.ReLU(inplace=True)
        self.dropout1m = nn.Dropout(self.p_dropout)

    def forward(self, x):
        x_color = x[:,:34]
        y = self.w1(x_color)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        

        ym = self.w1m(x)
        ym = self.batch_norm1m(ym)
        ym = self.relu1m(ym)
        ym = self.dropout1m(ym)

        y = torch.cat((y, ym), dim=1)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)

        return y


class SecondStage(nn.Module):
    def __init__(self, aux_input_size, linear_size, p_dropout=0.5):
        super().__init__()
        print(f"SecondStage,  size {aux_input_size}")
        self.aux_input_size = aux_input_size
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.aux_input_size , self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = TransformerModel(self.l_size, self.l_size, num_layers=1, num_heads=1, dropout=p_dropout)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
    
    def forward(self, x):
        
        x = self.w1(x)
        y = self.batch_norm1(x)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(x,y)
        if y.shape != x.shape:
            y.unsqueeze_(0)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class ModifiedLinearBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = TransformerModel(self.l_size, self.l_size, num_layers=1, num_heads=1, dropout=p_dropout)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
    def forward(self, x):
        
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(x,y)
        if y.shape != x.shape:
            y.unsqueeze_(0)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
    
    
class Lazaros(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, linear_size, num_stage=2):
        super().__init__()

        self.input_size = input_size
        self.linear_size = linear_size
        self.num_stage = num_stage
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ModifiedLinearBlock(self.linear_size, 0))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(self.linear_size, 1)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0)

    def forward(self, x):
        
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        
        #for i in range(self.num_stage):
        #    y = self.linear_stages[i](y)
        y = self.w2(y)
        

        #print(y)
        return y
    
    

class IdentityMapping(nn.Module):
    def __init__(self):
        super(IdentityMapping, self).__init__()
        self.linear = nn.Linear(17, 1)
        # Setting weight to 1 and bias to 0
        self.linear.weight.data.fill_(1)
        self.linear.bias.data.fill_(0)
        
        # Freezing the parameters of the frozen_part
        #for param in self.linear.parameters():
        #    param.requires_grad = False

    def forward(self, x):
        #print(f'self.linear.weight{self.linear.weight}, self.linear.bias{self.linear.bias}')
        return self.linear(x)

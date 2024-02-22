import torch.nn as nn
import torch

class MonolocoModel(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()
        print(f'MonolocoModel with {num_stage} stages')
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
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x[:,:34])
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y

class EarlyGuidance(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()
        print('EarlyGuicance')
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
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)


        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)

        
    def forward(self, x):
        # pre-processing
        #print(x.dtype)
        x_color = x[:,:34]
        #x_depth = x[:,34:]
        x_depth = x[:,34:] / 100
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
            
        y = self.w2(y)


        #pred_y = self.pre_w2(x_depth)
        #pred_y = self.batch_norm2(pred_y)
        #pred_y = self.relu2(pred_y)
        #pred_y = self.dropout2(pred_y)
        
        
        #print('y',torch.min(y),torch.max(y))
        #print('pred_y',torch.min(pred_y),torch.max(pred_y))
        #y = self.w2(y)
        #y = y + pred_y
        #y_empty = torch.zeros_like(y)
        #y = torch.cat((y, y_empty), dim=1)

        #y = self.w2_2_modified(y)


        return y

    
class MonolocoModelSparse(nn.Module):
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
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        #self.w2_modified = nn.Linear(self.linear_size, self.output_size)
        #self.w2_2_modified = nn.Linear(self.linear_size * 2, self.output_size)


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
        x_depth = x[:,34:] / 100
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

class MonolocoModelhumaNNMoD(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()
        print('MonolocoModelhumaNN_MODIFIED')
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
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2_2_modified = nn.Linear(self.linear_size * 2, self.output_size)

        self.pre_w2 = nn.Linear(17, self.linear_size)
        
        #self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.batch_norm2 = nn.BatchNorm1d(self.linear_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(self.p_dropout)
        #self.input_size = input_size
        
    def forward(self, x):
        # pre-processing
        #print(x.dtype)
        x_color = x[:,:34]
        #x_depth = x[:,34:]
        x_depth = x[:,34:] / 100
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
            
        #y = self.w2(y)


        pred_y = self.pre_w2(x_depth)
        pred_y = self.batch_norm2(pred_y)
        pred_y = self.relu2(pred_y)
        pred_y = self.dropout2(pred_y)
        
        
        #print('y',torch.min(y),torch.max(y))
        #print('pred_y',torch.min(pred_y),torch.max(pred_y))
        #y = self.w2(y)
        #y = y + pred_y
        y_empty = torch.zeros_like(y)
        y = torch.cat((y, y_empty), dim=1)

        y = self.w2_2_modified(y)


        return y

class MonolocoModelhumaNN(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()
        print('MonolocoModelhumaNN')
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
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        #self.w2 = nn.Linear(self.linear_size, self.output_size)
        #self.w2_modified = nn.Linear(self.linear_size, self.output_size)
        self.w2_2_modified = nn.Linear(self.linear_size * 2, self.output_size)


        self.pre_w2 = nn.Linear(17, self.linear_size)
        
        #self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.batch_norm2 = nn.BatchNorm1d(self.linear_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.p_dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(self.p_dropout)
        #self.input_size = input_size
        
    def forward(self, x):
        # pre-processing
        #print(x.dtype)
        x_color = x[:,:34]
        #x_depth = x[:,34:]
        x_depth = x[:,34:] / 100
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
            
        #y = self.w2(y)


        pred_y = self.pre_w2(x_depth)
        pred_y = self.batch_norm2(pred_y)
        pred_y = self.relu2(pred_y)
        pred_y = self.dropout2(pred_y)
        
        
        #print('y',torch.min(y),torch.max(y))
        #print('pred_y',torch.min(pred_y),torch.max(pred_y))
        #y = self.w2(y)
        y = y + pred_y
        y = torch.cat((y, pred_y), dim=1)

        y = self.w2_2_modified(y)


        return y

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
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
    
   
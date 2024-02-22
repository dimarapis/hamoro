from torchview import draw_graph
from monoloco.monoloco_model import MonolocoModel, MonolocoModelhumaNN,MonolocoModelhumaNNMoD, EarlyGuidance
from monoloco.humann_model import MyLinear
import torch
import graphviz
graphviz.set_jupyter_format('png')
from monoloco.humann_model import TestModel_3s,TestModel_3m,TestModel_3d,TestModel_3s_d



def network_layers():
    input_size = 34
    output_size = 2
    n_dropout=0
    p_dropout=0.2
    linear_size=256

    model = TestModel_3s_d(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                            output_size=output_size,num_stage=3)

    #model = MyLinear(linear_size=linear_size, p_dropout=p_dropout)

    batch_size = 16
    # device='meta' -> no memory is consumed for visualization
    model_graph = draw_graph(model, input_size=(1, 51), device='meta',save_graph=True,directory='.')
    #model_graph.visual_graph


#def network_activation():
#    model = torch.load('data/dimidata_test_filtered_fixed/monoloco_best_model.pth')

network_layers()
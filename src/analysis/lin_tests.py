import numpy as np
import pandas as pd
from torch.autograd.functional import jvp, vjp
import torch


def adj_test(x0, y0, model):
    pert_x = torch.randn_like(x0)
    pert_y = torch.randn_like(y0)

    _, pert_yy = jvp(model, x0, pert_x)
    _, pert_xx = vjp(model, x0, pert_y)

    d =  torch.allclose(torch.matmul(pert_xx.flatten(), pert_x.flatten()), torch.matmul(pert_yy.flatten(), pert_y.flatten() +0.00001))
    x_prod = torch.matmul(pert_xx.flatten(), pert_x.flatten()).item()
    y_prod = torch.matmul(pert_xx.flatten(), pert_x.flatten()).item()
    diff = abs(x_prod - y_prod)
    return x_prod, y_prod, diff
    
def tlm_test(x0, model, perturbation_scale_list):
    scale_list = []
    norm_perturbation_list = []
    
    for perturbation_size in perturbation_scale_list:
        pert_x = torch.randn_like(x0) * perturbation_size
        y, dy = jvp(model, x0, pert_x)
        y_pert_truth = model(x0 + pert_x)
        scale =  (y - y_pert_truth) / dy # must be 'one'        
        scale_list.append( torch.mean(abs(scale)).item()  )
        norm_perturbation_list.append( torch.sum(abs(pert_x)).item())
    return scale_list, norm_perturbation_list, scale.flatten().numpy()

def lin_test(x0, y0, model):
    pert_in = torch.randn_like(x0)
    pert_out = torch.randn_like(y0)

    _, pert_yy = jvp(model, x0, pert_in)
    _, pert_xx = vjp(model, x0, pert_out)

    _, pert_yy_2 = jvp(model, x0, 2 * pert_in)
    _, pert_xx_2 = vjp(model, x0, 2 * pert_out)
    
    diff_x = torch.mean(abs( 2 * pert_xx - pert_xx_2)).item()
    diff_y = torch.mean(abs( 2 * pert_yy - pert_yy_2)).item()
    return diff_x, diff_y


def main(data_loader, model, num_elements=0, elements=[], num_perturbations=1):
    perturbation_scale_list = []
    for j in [0, -1, -2, -3, -4, -5, -6, -7, -8]:
        for i in [5, 2, 1]:
            perturbation_scale_list.append(i* (10**j))

    assert(len(elements)> 0 or num_elements>0)
    L = len(data_loader)
    bs = data_loader.batch_size
        
    if len(elements) <= 0:
        elements = np.random.randint(0, L*bs, size = num_elements)
    elements.sort()
    print(elements)
    print(num_elements)
    
    transpose_diff_list = []
    lin_scale_list = []
    lin_tlm_diff_list = []
    lin_adj_diff_list = []
    last_scale_list = []
    
    for element in elements:
        batch_id   = element // data_loader.batch_size
        element_id = element % data_loader.batch_size
        x, y = data_loader[batch_id]
        x0 = x[[element_id]]
        y0 = y[[element_id]]
        for _ in range(num_perturbations):
            x_prod, y_prod, tlm_adj_diff = adj_test(x0, y0, model)
            scale_list, norm_perturbation_list, last_scale_el = tlm_test(x0, model, perturbation_scale_list)        
            adj_diff_x, tlm_diff_y = lin_test(x0, y0, model)
            #########################################
            transpose_diff_list.append(tlm_adj_diff)
            lin_tlm_diff_list.append(tlm_diff_y)
            lin_adj_diff_list.append(adj_diff_x)
            lin_scale_list.append(scale_list)
            last_scale_list.append(last_scale_el)
  
    return {'transpose' : transpose_diff_list, 
            'lin_tlm' : lin_tlm_diff_list, 
            'lin_adj' : lin_adj_diff_list, 
            'tlm_mean_error' : lin_scale_list,
            'tlm_pert':perturbation_scale_list,
            'tlm_last_error':last_scale_list
            }

import matplotlib.pyplot as plt

def show_result(tlm_adj_diff, lin_tlm_diff_list, lin_adj_diff_list, scale_list, last_scale_list, perturbation_scale_list):
    f_transpoe = tranpose_fig(tlm_adj_diff)
    f_lin = lin_fig(lin_tlm_diff_list, lin_adj_diff_list)
    f_tlm_plot = lin_plot_fig(perturbation_scale_list, scale_list)
    f_tlm_hist = lin_hist_fig(last_scale_list)
    return f_transpoe, f_lin, f_tlm_plot, f_tlm_hist
        
def tranpose_fig(tlm_adj_diff):
    f = plt.figure()
    plt.hist(tlm_adj_diff)
    plt.title('Tranpose Test')
    return f


def lin_fig(lin_tlm_diff_list, lin_adj_diff_list):
    f = plt.figure()
    ax = f.add_subplot(1,2,1)
    ax.hist(lin_adj_diff_list)
    ax.set_title('Adj Lin Test')
    ax = f.add_subplot(1,2,2)
    ax.hist(lin_tlm_diff_list)
    ax.set_title('Tlm Lin Test')
    return f

def lin_plot_fig(perturbation_scale_list, scale_list):
    f = plt.figure()
    for mean_scale_list in scale_list:
        plt.plot(perturbation_scale_list, mean_scale_list )
    plt.xlabel('perturbation scale')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    return f

def lin_hist_fig(last_scale_list):
    f = plt.figure()
    for data in last_scale_list:
        plt.hist(data) 
    return f
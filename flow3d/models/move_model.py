import torch
import torch.nn as nn
import math

import pypose as pp

from flow3d.models.utils.spline_utils import SE3_to_se3, se3_to_SE3,cubic_bspline_interpolation,linear_interpolation,linear_interpolation_mid
import functools
import numpy as np


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

embed_P_fn, P_input_ch = get_embedder(5, 6)     # 63
embed_time_fn, time_input_ch = get_embedder(5, 1)     # 63


class MoveModel(nn.Module):
    def __init__(self, num_fg, camera_mode='linear'):
        super().__init__()
        self.slope = 0.01
        W = 32

        self.camera_mode = camera_mode
        
        self.RT_main = nn.Sequential(
                nn.Linear(P_input_ch, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(W*2, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(W*2, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(W*2, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(W*2, W*2),
            )

        self.RT_head0 = nn.Sequential(
                nn.Linear(W*2, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(2*W, 6)
            )
            
        self.RT_head1 = nn.Sequential(
                nn.Linear(W*2, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(2*W, 6)
            )

        self.time_params = torch.nn.Parameter(torch.ones(1, 8, requires_grad=True), requires_grad=True)
        self.time_params.data.fill_(0.5)
        self.relu = nn.ReLU()                   
        self.zero_initialize() 


    def zero_initialize(self,):
        nn.init.constant_(self.RT_head0[-1].weight, 0.)
        nn.init.constant_(self.RT_head0[-1].bias, 0.)

        nn.init.constant_(self.RT_head1[-1].weight, 0.)
        nn.init.constant_(self.RT_head1[-1].bias, 0.)

    
    def forward(self, R, T,  time, stage="second"):
        RT = self.preprocessPose(R, T)
        RT = RT.unsqueeze(0)  

        x = embed_P_fn(RT)  
        x = self.RT_main(x)
        detaRT0 = self.RT_head0(x)
        detaRT1 = self.RT_head1(x)

        if stage == "first":
            deltaT0 = torch.zeros(1).to(RT.device)
            deltaT1 = torch.zeros(1).to(RT.device)
        else:
            index = int(time)
            if index <= 0 or index >= self.time_params.shape[-1] -1:
                deltaT0 = torch.zeros_like(self.time_params[:, 0])
                deltaT1 = torch.zeros_like(self.time_params[:, 0])
            else:
                deltaT = self.time_params[:, index]
                deltaT = self.relu(deltaT).clamp(0.1, 0.9)
                deltaT0 = deltaT * -1.
                deltaT1 = deltaT * 1.

        return detaRT0, detaRT1,  deltaT0, deltaT1 
        

    def forward_start_end_mid(self, info, num_cameras=10, mode='uniform', stage="second"):
        R = info['R']
        T = info['T']
        time = info["timestep"]

        RT_start, RT_end, time_start, time_end = self.forward(R, T, time, stage=stage)
    
        RTs = self._interpolate(pp.se3(RT_start).Exp(), pp.se3(RT_end).Exp(), num_cameras=num_cameras, mode=mode)   # N 7
        RTs = RTs.Log()    #  N 6
        RTs = self.postprocessPose(RTs).squeeze(0)  # N 3 4

        # # interpolate time
        num_fg = time_start.shape[0]
        time_start = time_start.unsqueeze(-1).repeat(1, num_cameras)
        time_end = time_end.unsqueeze(-1).repeat(1, num_cameras)
        weights = (torch.arange(num_cameras) / (num_cameras-1)).to(RTs.device)
        weights = weights.unsqueeze(0).repeat(num_fg, 1)
        times = (time_start+time) * (1. - weights) + (time_end+time) * weights
        times = times.reshape(num_fg, num_cameras)  
        if mode == "uniform":
            times = times
        elif mode == "mid":
            times = times[:, (num_cameras//2):(num_cameras // 2 +1)]
        elif mode == "start":
            times = times[:, 0:1]
        elif mode == "end":
            times = times[:, num_cameras-1:]
        deltaT = torch.abs(time_end[:, num_cameras-1:])
        return RTs, times, deltaT

    def _interpolate(self, RT_start, RT_end, num_cameras, mode="uniform"):
        pp_start = pp.SE3(RT_start.unsqueeze(1)) 
        pp_end = pp.SE3(RT_end.unsqueeze(1))
        camera_opt = torch.cat([pp_start, pp_end], 1)
        if mode == "uniform":
            u = torch.linspace(start=0,end=1,steps=num_cameras,device=RT_start.device,
            )
            if self.camera_mode == "linear":
                return linear_interpolation(camera_opt, u)
            elif self.camera_mode == "cubic":
                return cubic_bspline_interpolation(camera_opt, u)
        elif mode == "mid":
            if self.camera_mode == "linear":
                return linear_interpolation_mid(camera_opt)
            elif self.camera_mode == "cubic":
                return cubic_bspline_interpolation(
                    camera_opt,
                    torch.tensor([0.5], device=camera_opt.device)
                ).squeeze(1)
        elif mode == "start":
            if self.camera_mode == "linear":
                return camera_opt[..., 0, :]
            elif self.camera_mode == "cubic":
                return cubic_bspline_interpolation(
                    camera_opt,
                    torch.tensor([0.0], device=camera_opt.device)
                ).squeeze(1)
        elif mode == "end":
            if self.camera_mode == "linear":
                return camera_opt[..., 1, :]
            elif self.camera_mode == "cubic":
                return cubic_bspline_interpolation(
                    camera_opt,
                    torch.tensor([1.0], device=camera_opt.device)
                ).squeeze(1)
        else:
            assert False, "_interpolate mode is uniform or mid!"

    def preprocessPose(self, R, T):
         blury_pose = torch.cat([R, T], axis=-1)
         RT = SE3_to_se3(blury_pose)  
         return RT
    
    def postprocessPose(self, RT):
        RT = se3_to_SE3(RT)
        return RT
    

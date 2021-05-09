import torch
import torchvision.utils as vutils
import numpy as np
from ..data.utils import decode_quantize
from connectomics.model.utils import SplitActivation

__all__ = [
    'Visualizer'
]

class Visualizer(object):
    """TensorboardX visualizer for displaying loss, learning rate and predictions
    at training time.
    """
    def __init__(self, cfg, vis_opt=0, N=16):
        self.cfg = cfg
        self.act = SplitActivation.build_from_cfg(cfg, do_cat=False)
        self.vis_opt = vis_opt
        self.N = N # default maximum number of sections to show
        self.N_ind = None

        self.semantic_colors = {}
        for topt in self.cfg.MODEL.TARGET_OPT:
            if topt[0] == '9':
                channels = int(topt.split('-')[1])
                colors = [torch.rand(3) for _ in range(channels)]
                colors[0] = torch.zeros(3) # make background black
                self.semantic_colors[topt] = torch.stack(colors, 0)

    def visualize(self, volume, label, output, weight, iter_total, writer, volume_aug_1= None, 
                        volume_aug_2= None, output_aug_1 = None, output_aug_2 = None, volume_track = None, output_track = None, volume_aug_track = None, output_aug_track = None, val=False):
        # split the prediction into chunks along the channel dimension
        output = self.act(output)
        assert len(output) == len(label)

        if self.cfg.MODEL.UDA:
            output_aug_1 = self.act(output_aug_1)
            output_aug_2 = self.act(output_aug_2)
            output_track = self.act(output_track)
            output_aug_track = self.act(output_aug_track)
            
            assert len(output_aug_1) == len(label)
            assert len(output_aug_2) == len(label)
            assert len(output_track) == len(label)
            assert len(output_aug_track) == len(label)
          
        for idx in range(len(self.cfg.MODEL.TARGET_OPT)):
            topt = self.cfg.MODEL.TARGET_OPT[idx]
            if topt[0] == '9':
                output[idx] = self.get_semantic_map(output[idx], topt)
                label[idx] = self.get_semantic_map(label[idx], topt, argmax=False)
                if self.cfg.MODEL.UDA:
                    output_aug_1[idx] = self.get_semantic_map(output_aug_1[idx], topt)
                    output_aug_2[idx] = self.get_semantic_map(output_aug_2[idx], topt)
                    output_track[idx] = self.get_semantic_map(output_track[idx], topt)
                    output_aug_track[idx] = self.get_semantic_map(output_aug_track[idx], topt)

            if topt[0] == '5':
                output[idx] = decode_quantize(output[idx], mode='max').unsqueeze(1)
                temp_label = label[idx].copy().astype(np.float32)[:, np.newaxis]
                label[idx] = temp_label / temp_label.max() + 1e-6
                if self.cfg.MODEL.UDA:
                    output_aug_1[idx] = decode_quantize(output_aug_1[idx], mode='max') #.unsqueeze(1)
                    output_aug_2[idx] = decode_quantize(output_aug_2[idx], mode='max') #.unsqueeze(1)
                    output_track[idx] = decode_quantize(output_track[idx], mode='max') #.unsqueeze(1)
                    output_aug_track[idx] = decode_quantize(output_aug_track[idx], mode='max') #.unsqueeze(1)
                    # temp_label = output_aug_1[idx].copy().astype(np.float32)[:, np.newaxis]
                    # label[idx] = temp_label / temp_label.max() + 1e-6

            RGB = (topt[0] in ['1', '2', '9'])
            vis_name = self.cfg.MODEL.TARGET_OPT[idx] + '_' + str(idx)
            if val: vis_name = vis_name + '_Val'
            if isinstance(label[idx], (np.ndarray, np.generic)):
                label[idx] = torch.from_numpy(label[idx])
            
            # if self.cfg.MODEL.UDA:
            #     if isinstance(output_aug_1[idx], (np.ndarray, np.generic)):
            #         output_aug_1[idx] = torch.from_numpy(output_aug_1[idx])
            #     if isinstance(output_aug_2[idx], (np.ndarray, np.generic)):
            #         output_aug_2[idx] = torch.from_numpy(output_aug_2[idx])
            
            weight_maps = {}
            for j, wopt in enumerate(self.cfg.MODEL.WEIGHT_OPT[idx]):
                if wopt != '0':
                    w_name = vis_name + '_' + wopt
                    weight_maps[w_name] = weight[idx][j]
            
            if self.cfg.MODEL.UDA:
                self.visualize_consecutive(volume, label[idx], output[idx], weight_maps, iter_total, 
                                       writer, volume_aug_1, volume_aug_2, volume_track, volume_aug_track,  output_aug_1[idx], output_aug_2[idx], output_track[idx], output_aug_track[idx], RGB=RGB, vis_name=vis_name)
            else:
                self.visualize_consecutive(volume, label[idx], output[idx], weight_maps, iter_total, 
                                       writer, RGB=RGB, vis_name=vis_name)

    def visualize_consecutive(self, volume, label, output, weight_maps, iteration, 
                              writer, volume_aug_1= None, volume_aug_2= None, volume_track = None, volume_aug_track = None, 
                              output_aug_1= None, output_aug_2= None, output_track = None, output_aug_track = None, RGB=False, vis_name='0_0'):

        volume, label, output, weight_maps, volume_aug_1, volume_aug_2, volume_track, volume_aug_track, output_aug_1, output_aug_2, output_track, output_aug_track = self.prepare_data(volume, label, output, weight_maps, volume_aug_1, volume_aug_2, volume_track, volume_aug_track, output_aug_1, output_aug_2, output_track, output_aug_track)
        
        sz = volume.size() # z,c,y,x
        canvas = []
        volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])

        canvas.append(volume_visual)

        if RGB:
            output_visual = [output.detach().cpu()]
            label_visual = [label.detach().cpu()]
            if self.cfg.MODEL.UDA:
                output_aug_1_visual = [output_aug_1.detach().cpu()]
                output_aug_2_visual = [output_aug_2.detach().cpu()]
                output_track_visual = [output_track.detach().cpu()]
                output_aug_track_visual = [output_aug_track.detach().cpu()]
        else:
            output_visual = [self.vol_reshape(output[:,i], sz) for i in range(sz[1])]
            label_visual = [self.vol_reshape(label[:,i], sz) for i in range(sz[1])]
            if self.cfg.MODEL.UDA:
                output_aug_1_visual = [self.vol_reshape(output_aug_1[:,i], sz) for i in range(sz[1])]
                output_aug_2_visual = [self.vol_reshape(output_aug_2[:,i], sz) for i in range(sz[1])]
                output_track_visual = [self.vol_reshape(output_track[:,i], sz) for i in range(sz[1])]
                output_aug_track_visual = [self.vol_reshape(output_aug_track[:,i], sz) for i in range(sz[1])]

        weight_visual = []
        for key in weight_maps.keys():
            weight_visual.append(weight_maps[key].detach().cpu().expand(sz[0],3,sz[2],sz[3]))

        canvas = canvas + output_visual + label_visual + weight_visual
        canvas_merge = torch.cat(canvas, 0)
        canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)
        writer.add_image('Consecutive_%s' % vis_name, canvas_show, iteration)

        
        if self.cfg.MODEL.UDA:
            canvas_uda = []
            volume_aug_1_visual = volume_aug_1.detach().cpu().expand(sz[0],3,sz[2],sz[3])
            volume_aug_2_visual = volume_aug_2.detach().cpu().expand(sz[0],3,sz[2],sz[3])
            output_aug_1_visual = torch.cat(output_aug_1_visual)
            output_aug_2_visual = torch.cat(output_aug_2_visual)
            for vol1, vol2, out1, out2 in zip(volume_aug_1_visual, volume_aug_2_visual, output_aug_1_visual, output_aug_2_visual):
                canvas_uda.append(torch.unsqueeze(vol1, 0))
                canvas_uda.append(torch.unsqueeze(out1, 0))
                canvas_uda.append(torch.unsqueeze(vol2, 0))
                canvas_uda.append(torch.unsqueeze(out2, 0))

            # canvas_uda = canvas_uda + output_aug_1_visual + output_aug_2_visual
            canvas_merge = torch.cat(canvas_uda, 0)
            canvas_show_uda = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)
            writer.add_image('Consecutive_uda%s' % vis_name, canvas_show_uda, iteration)

            canvas_tracker = []
            volume_track_visual = volume_track.detach().cpu().expand(sz[0],3,sz[2],sz[3])
            volume_aug_track_visual = volume_aug_track.detach().cpu().expand(sz[0],3,sz[2],sz[3])

            output_track_visual = torch.cat(output_track_visual)
            output_aug_track_visual = torch.cat(output_aug_track_visual)

            for vol1, out1, vol2, out2 in zip(volume_track_visual, output_track_visual, volume_aug_track_visual, output_aug_track_visual):
                canvas_tracker.append(torch.unsqueeze(vol1, 0))
                canvas_tracker.append(torch.unsqueeze(out1, 0))
                canvas_tracker.append(torch.unsqueeze(vol2, 0))
                canvas_tracker.append(torch.unsqueeze(out2, 0))

            canvas_merge = torch.cat(canvas_tracker, 0)
            canvas_show_tracker = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)
            writer.add_image('Target_tracker%s' % vis_name, canvas_show_tracker, iteration)


    def prepare_data(self, volume, label, output, weight_maps, volume_aug_1, volume_aug_2, volume_track, volume_aug_track, output_aug_1, output_aug_2, output_track, output_aug_track):
        ndim = volume.ndim
        assert ndim in [4, 5] 
        is_3d = (ndim == 5)

        volume = self.permute_truncate(volume, is_3d)
        label = self.permute_truncate(label, is_3d)
        output = self.permute_truncate(output, is_3d)

        if self.cfg.MODEL.UDA:
            volume_aug_1 = self.permute_truncate(volume_aug_1, is_3d)
            volume_aug_2 = self.permute_truncate(volume_aug_2, is_3d)
            output_aug_1 = self.permute_truncate(output_aug_1, is_3d)
            output_aug_2 = self.permute_truncate(output_aug_2, is_3d)
            volume_track = self.permute_truncate(volume_track, is_3d)
            volume_aug_track = self.permute_truncate(volume_aug_track, is_3d)
            output_track = self.permute_truncate(output_track, is_3d)
            output_aug_track = self.permute_truncate(output_aug_track, is_3d)


        for key in weight_maps.keys():
            weight_maps[key] = self.permute_truncate(weight_maps[key], is_3d)

        return volume, label, output, weight_maps, volume_aug_1, volume_aug_2, volume_track, volume_aug_track, output_aug_1, output_aug_2, output_track, output_aug_track

    def permute_truncate(self, data, is_3d=False):
        if is_3d:
            data = data[0].permute(1,0,2,3)
        high = min(data.size()[0], self.N)
        return data[:high]

    def get_semantic_map(self, output, topt, argmax=True):
        if isinstance(output, (np.ndarray, np.generic)):
            output = torch.from_numpy(output)
        # output shape: BCDHW or BCHW
        if argmax:
            output = torch.argmax(output, 1)
        pred = self.semantic_colors[topt][output] 
        if len(pred.size()) == 4:   # 2D Inputs
            pred = pred.permute(0,3,1,2)
        elif len(pred.size()) == 5: # 3D Inputs
            pred = pred.permute(0,4,1,2,3)
        
        return pred

    def vol_reshape(self, vol, sz):
        vol = vol.detach().cpu().unsqueeze(1)
        return vol.expand(sz[0],3,sz[2],sz[3])

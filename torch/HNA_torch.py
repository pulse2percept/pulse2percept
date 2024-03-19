import torch
import numpy as np

# from HNA tensorflow implementation
class UniversalBiphasicAxonMapModule(torch.nn.Module):
    def __init__(self, p2pmodel, implant, activity_regularizer=None, clip=None, amp_cutoff=True, **kwargs):
        super().__init__()

        dtype = torch.get_default_dtype()

        # p2pmodel.min_ax_sensitivity = 0.2 don't here
        bundles = p2pmodel.grow_axon_bundles() # 763 [[20-300]x,y]
        axons = p2pmodel.find_closest_axon(bundles) # 2401 [[20-300]x,y]
        if type(axons) != list:
            axons = [axons]
        axon_contrib = calc_axon_sensitivity(p2pmodel, axons, pad=True)
        axon_contrib = torch.tensor(axon_contrib, dtype=dtype) # 2401 pix, 118 l_ax, 3 (x,y,sens)

        self.register_buffer("axon_contrib", axon_contrib)


        # Get implant parameters
        # self.n_elecs = len(implant.electrodes)
        self.elec_x = torch.tensor([implant[e].x for e in implant.electrodes], dtype=dtype)
        self.elec_y = torch.tensor([implant[e].y for e in implant.electrodes], dtype=dtype)

        d2_el = (self.axon_contrib[:, :, 0, None] - self.elec_x)**2 + \
                (self.axon_contrib[:, :, 1, None] - self.elec_y)**2 # 2401, 118, 225

        self.register_buffer("d2_el", d2_el)

        self.clip = False
        if isinstance(clip, tuple):
            self.clip = True
            self.clipmin = clip[0]
            self.clipmax = clip[1]

        self.amp_cutoff = amp_cutoff
        self.percept_shape = p2pmodel.grid.shape
        self.thresh_percept = p2pmodel.thresh_percept

    def forward(self, inputs, like_jax=False):
        freq = inputs[0][:, :, 0]
        amp = inputs[0][:, :, 1]
        pdur = inputs[0][:, :, 2]

        rho = inputs[1][:, 0][:, None]
        axlambda = inputs[1][:, 1][:, None]
        a0 = inputs[1][:, 2][:, None]
        a1 = inputs[1][:, 3][:, None]
        a2 = inputs[1][:, 4][:, None]
        a3 = inputs[1][:, 5][:, None]
        a4 = inputs[1][:, 6][:, None]
        a5 = inputs[1][:, 7][:, None]
        a6 = inputs[1][:, 8][:, None]
        a7 = inputs[1][:, 9][:, None]
        a8 = inputs[1][:, 10][:, None]
        a9 = inputs[1][:, 11][:, None]

        scaled_amps = (a1 + a0*pdur) * amp

        # bright
        F_bright = a2 * scaled_amps + a3 * freq
        if self.amp_cutoff:
            F_bright = torch.where(scaled_amps > 0.25, F_bright, torch.zeros_like(F_bright))

        if not like_jax: # like pyx impl.
            F_bright = torch.where(amp > 0, F_bright, torch.zeros_like(F_bright))

        # size
        min_f_size = 10**2 / (rho**2)
        F_size = a5 * scaled_amps + a6
        F_size = torch.maximum(F_size, min_f_size)

        # streak
        min_f_streak = 10**2 / (axlambda ** 2)
        F_streak = a9 - a7 * pdur ** a8
        F_streak = torch.maximum(F_streak, min_f_streak)

        # eparams = torch.stack([F_bright, F_size, F_streak], axis=2) # 1, 225, 3

        # apply axon map
        intensities =   (
                        F_bright[:, None, None, :] * # 1, 1, 1, 225
                        torch.exp(
                                    -self.d2_el[None, :, :, :] / # dist2el 1, 2401, 118, 225
                                    (2. * rho**2 * F_size)[:, None, None, :] # 1, 1, 1, 225
                                    + # contribution of each electode to each axon segement of each
                                      # pixel by distance of segemnt to electrode
                                    self.axon_contrib[None, :, :, 2, None] / # sens 1, 2401, 118, 1
                                    (axlambda** 2 * F_streak)[:, None, None, :] # 1, 1, 1 , 225
                                      # contribution of each electode to each axon segement of each
                                      # pixel by sensitivity, which is scaled by axon distance
                                 ) # 1, 2401, 118, 225, scaling between 0, 1
                        ) # 1, 2401, 118, 225

        # after summing up...
        intensities = torch.max(torch.sum(intensities, axis=-1), axis=-1).values # sum over electrodes, max over segments
        intensities = torch.where(intensities > self.thresh_percept, intensities, torch.zeros_like(intensities))
        if self.clip:
            intensities = torch.clamp(intensities, self.clipmin, self.clipmax)

        batched_percept_shape = tuple([-1] + list(self.percept_shape))
        intensities = intensities.reshape(batched_percept_shape)
        return intensities

# static model
class AxonMapSpatialModule(torch.nn.Module):
    def __init__(self, p2pmodel, implant, activity_regularizer=None, clip=None, amp_cutoff=True, **kwargs):
        super().__init__()

        dtype = torch.get_default_dtype()

        # p2pmodel.min_ax_sensitivity = 0.2
        bundles = p2pmodel.grow_axon_bundles() # 763 [[20-300]x,y]
        # ok beyeler2019

        axons = p2pmodel.find_closest_axon(bundles) # 2401 [[20-300]x,y]
        # ok beyeler2019

        if type(axons) != list:
            axons = [axons]
        axon_contrib = calc_axon_sensitivity(p2pmodel, axons, pad=True) # similar beyeler2019 without axlambda
        axon_contrib = torch.tensor(axon_contrib, dtype=dtype) # 2401 pix, 118 l_ax, 3 (x,y,sens)

        self.register_buffer("axon_contrib", axon_contrib)


        # Get implant parameters
        # self.n_elecs = len(implant.electrodes)
        self.elec_x = torch.tensor([implant[e].x for e in implant.electrodes], dtype=dtype)
        self.elec_y = torch.tensor([implant[e].y for e in implant.electrodes], dtype=dtype)

        d2_el = (self.axon_contrib[:, :, 0, None] - self.elec_x)**2 + \
                (self.axon_contrib[:, :, 1, None] - self.elec_y)**2 # 2401, 118, 225

        self.register_buffer("d2_el", d2_el)

        self.clip = False
        if isinstance(clip, tuple):
            self.clip = True
            self.clipmin = clip[0]
            self.clipmax = clip[1]

        self.amp_cutoff = amp_cutoff
        self.percept_shape = p2pmodel.grid.shape
        self.thresh_percept = p2pmodel.thresh_percept

    def forward(self, inputs):
        amp = inputs[0][:, :]

        rho = inputs[1][:, 0][:, None]
        axlambda = inputs[1][:, 1][:, None]

        # apply axon map
        intensities =   (
                        amp[:, None, None, :] * # 1, 1, 1, 225
                        torch.exp(   # gauss
                                    -self.d2_el[None, :, :, :] / # dist2el 1, 2401, 118, 225
                                    (2. * rho**2)[:, None, None, :] # 1, 1, 1, 225
                                    + # contribution of each electode to each axon segement of each
                                      # pixel by distance of segemnt to electrode
                                    self.axon_contrib[None, :, :, 2, None] / # sens 1, 2401, 118, 1
                                    (axlambda**2)[:, None, None, :] # 1, 1, 1 , 225
                                      # contribution of each electode to each axon segement of each
                                      # pixel by sensitivity, which is scaled by distance along axon
                                 ) # 1, 2401, 118, 225, scaling between 0, 1
                        ) # 1, 2401, 118, 225



        # after summing up...
        intensities_per_axon = torch.sum(intensities, axis=-1)
        intensities = torch.take_along_dim(
            intensities_per_axon, intensities_per_axon.abs().max(-1, keepdim=True).indices, dim=-1).squeeze(-1)

        intensities = torch.where(intensities.abs() > self.thresh_percept, intensities, torch.zeros_like(intensities))


        if self.clip:
            intensities = torch.clamp(intensities, self.clipmin, self.clipmax)

        batched_percept_shape = tuple([-1] + list(self.percept_shape))
        intensities = intensities.reshape(batched_percept_shape)
        return intensities


def calc_axon_sensitivity(p2pmodel, bundles, pad=False):
    xyret = np.column_stack((p2pmodel.grid.xret.ravel(), p2pmodel.grid.yret.ravel()))
    # Only include axon segments that are < `max_d2` from the soma. These
    # axon segments will have `sensitivity` > `self.min_ax_sensitivity`:
    max_d2 = -2.0 * 3000 ** 2 * np.log(p2pmodel.min_ax_sensitivity) # axlambda
    axon_contrib = []
    for xy, bundle in zip(xyret, bundles):
        idx = np.argmin((bundle[:, 0] - xy[0]) ** 2 +
                        (bundle[:, 1] - xy[1]) ** 2)
        # Cut off the part of the fiber that goes beyond the soma:
        axon = np.flipud(bundle[0: idx + 1, :])
        # Add the exact location of the soma:
        axon = np.concatenate((xy.reshape((1, -1)), axon), axis=0)
        # For every axon segment, calculate distance from soma by
        # summing up the individual distances between neighboring axon
        # segments (by "walking along the axon"):
        d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                               np.diff(axon[:, 1], axis=0) ** 2)) ** 2
        idx_d2 = d2 < max_d2
        sensitivity = -d2[idx_d2] / 2 # axlambda
        # sensitivity = np.exp(-d2[idx_d2] / (2.0 * self.axlambda ** 2))
        idx_d2 = np.concatenate(([False], idx_d2))
        contrib = np.column_stack((axon[idx_d2, :], sensitivity)) # l_axon, 3 (x,y,sens)
        axon_contrib.append(contrib)

    if pad:
        # pad to length of longest axon
        axon_length = max([len(axon) for axon in axon_contrib])
        axon_sensitivities = np.zeros((len(axon_contrib), axon_length, 3)) # pix, l_ax, 3
        for i, axon in enumerate(axon_contrib):
            original_len = len(axon)
            if original_len >= axon_length:
                axon_sensitivities[i] = axon[:axon_length]
            elif original_len != 0:
                axon_sensitivities[i, :original_len] = axon
                axon_sensitivities[i, original_len:] = axon[-1] # repeat last til end

        del axon_contrib
        return axon_sensitivities
    else:
        return axon_contrib

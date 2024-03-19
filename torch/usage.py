import torch
import pulse2percept as p2p
import HNA_torch
import math

####
# setup
####
p2pmodel = p2p.models.BiphasicAxonMapModel(
        xrange=(-12, 12), yrange=(-12, 12),
        xystep=.9, a0=0, a1=1,
        min_ax_sensitivity=0.2,
        n_ax_segments=300)
p2pmodel.build()

impl_s = 9, 9
implant = p2p.implants.ElectrodeGrid(impl_s, 800, x=-0, y=0, z=0, rot=0)

####
# phi
####
def getphi(p2pmodel):
    attr = ['rho', 'axlambda', 'a0','a1','a2','a3','a4','a5','a6','a7','a8','a9']
    return {a: getattr(p2pmodel, a) for a in attr}
phi_ = getphi(p2pmodel)
phi_['rho'] = 400
phi_['axlambda'] = 1550

def phitens(phi):
    return torch.tensor(list(phi.values()))
phi = phitens(phi_)

####
# Axon Map Model
####
torchmod = HNA_torch.AxonMapSpatialModule(p2pmodel, implant, amp_cutoff=False)
stim = torch.rand(1, *impl_s)

stim = stim.to('cpu')
phi = phi.to('cpu')
torchmod.to('cpu')

pcpt = torchmod([stim.reshape(-1, math.prod(impl_s)), phi.repeat(1, 1)])

print(pcpt)

####
# Biphasic Axon Map Model
####
torchmod = HNA_torch.UniversalBiphasicAxonMapModule(p2pmodel, implant, amp_cutoff=False)
stim = torch.rand(1, *impl_s, 3)

stim = stim.to('cpu')
phi = phi.to('cpu')
torchmod.to('cpu')

pcpt = torchmod([stim.reshape(-1, math.prod(impl_s), 3), phi.repeat(1, 1)])

print(pcpt)

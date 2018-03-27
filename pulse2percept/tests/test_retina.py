import numpy as np
import scipy.special as ss
import scipy.optimize as scpo
import numpy.testing as npt
import pytest
import logging

from .. import retina
from .. import implants
from .. import stimuli
from .. import utils


class LegacyNanduri2012(retina.Nanduri2012):
    """Preserve old implementation to make sure Cython model runs correctly"""

    def __init__(self, **kwargs):
        # Set default values of keyword arguments
        self.tau1 = 0.42 / 1000
        self.tau2 = 45.25 / 1000
        self.tau3 = 26.25 / 1000
        self.eps = 8.73
        self.asymptote = 14.0
        self.slope = 3.0
        self.shift = 16.0

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

        _, self.gamma1 = utils.gamma(1, self.tau1, self.tsample)
        _, self.gamma2 = utils.gamma(1, self.tau2, self.tsample)
        _, self.gamma3 = utils.gamma(3, self.tau3, self.tsample)

    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Nanduri model cascade

        Parameters
        ----------
        in_arr: array - like
            A 2D array specifying the effective current values
            at a particular spatial location(pixel); one value
            per retinal layer and electrode.
            Dimensions: <  # layers x #electrodes>
        pt_list: list
            List of pulse train 'data' containers.
            Dimensions: <  # electrodes x #time points>
        layers: list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        use_jit: bool
            If True, applies just - in-time(JIT) compilation to
            expensive computations for additional speed - up
            (requires Numba).
        """
        if 'INL' in layers:
            raise ValueError("The Nanduri2012 model does not support an inner "
                             "nuclear layer.")

        # Although the paper says to use cathodic-first, the code only
        # reproduces if we use what we now call anodic-first. So flip the sign
        # on the stimulus here:
        b1 = -self.calc_layer_current(in_arr, pt_list)

        # Fast response
        b2 = self.tsample * utils.conv(b1, self.gamma1, mode='full',
                                       method='sparse',
                                       use_jit=use_jit)[:b1.size]

        # Charge accumulation
        ca = self.tsample * np.cumsum(np.maximum(0, b1))
        ca = self.tsample * utils.conv(ca, self.gamma2, mode='full',
                                       method='fft')[:b1.size]
        b3 = np.maximum(0, b2 - self.eps * ca)

        # Stationary nonlinearity
        b3max = b3.max()
        sigmoid = ss.expit((b3max - self.shift) / self.slope)
        b4 = b3 / b3max * sigmoid * self.asymptote

        # Slow response
        b5 = self.tsample * utils.conv(b4, self.gamma3, mode='full',
                                       method='fft')[:b1.size]

        return utils.TimeSeries(self.tsample, b5)


def test_Grid():
    # Invalid calls
    with pytest.raises(ValueError):
        grid = retina.Grid(x_range=1, n_axons=1)
    with pytest.raises(ValueError):
        grid = retina.Grid(x_range=(1.0, 0.0), n_axons=1)
    with pytest.raises(ValueError):
        grid = retina.Grid(y_range=1, n_axons=1)
    with pytest.raises(ValueError):
        grid = retina.Grid(y_range=(1.0, 0.0), n_axons=1)
    for n_axons in [-1, 0]:
        with pytest.raises(ValueError):
            grid = retina.Grid(n_axons=n_axons)
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            retina.Grid(n_rho=n_rho)
    for lophi in [-200.0, 90.0, 200.0]:
        with pytest.raises(ValueError):
            retina.Grid(phi_range=(lophi, 85.0))
    for hiphi in [-200.0, 90.0, 200.0]:
        with pytest.raises(ValueError):
            retina.Grid(phi_range=(95.0, hiphi))

    # Verify size of axon bundles
    for n_axons in [3, 5, 10]:
        grid = retina.Grid(x_range=(0, 0), y_range=(0, 0), n_axons=n_axons,
                           save_data=False)
        # Number of axon bundles given by `n_axons`
        npt.assert_equal(len(grid.axon_bundles), n_axons)
        # Number of axons given by number of pixels on grid
        npt.assert_equal(len(grid.axons), 1)

    # Verify file


def test_BaseModel():
    # Cannot instantiate abstract class
    with pytest.raises(TypeError):
        tm = retina.BaseModel(0.01)

    # Child class must provide `model_cascade()`
    class Incomplete(retina.BaseModel):
        pass
    with pytest.raises(TypeError):
        tm = Incomplete()

    # A complete class
    class Complete(retina.BaseModel):

        def model_cascade(self, inval):
            return inval

    tm = Complete(tsample=0.1)
    npt.assert_equal(tm.tsample, 0.1)
    npt.assert_equal(tm.model_cascade(2.4), 2.4)


def test_TemporalModel():
    tsample = 0.01 / 1000

    # Assume 4 electrodes, each getting some stimulation
    pts = [stimuli.PulseTrain(tsample=tsample, dur=0.1)] * 4
    ptrain_data = [pt.data for pt in pts]

    # For each of these 4 electrodes, we have two effective current values:
    # one for the ganglion cell layer, one for the bipolar cell layer
    ecs_item = np.random.rand(2, 4)

    # What we want is the effective current per layer, for all electrodes
    # added up.
    # We do this for different layer configurations:
    for layers in [['INL'], ['GCL'], ['GCL', 'INL']]:
        ecm_by_hand = np.zeros((2, ptrain_data[0].size))
        if 'INL' in layers:
            for curr, pt in zip(ecs_item[0, :], ptrain_data):
                ecm_by_hand[0, :] += curr * pt
        if 'GCL' in layers:
            for curr, pt in zip(ecs_item[1, :], ptrain_data):
                ecm_by_hand[1, :] += curr * pt

        # And that should be the same as `calc_layer_current`:
        tm = retina.TemporalModel(tsample=tsample)
        ecm = tm.calc_layer_current(ecs_item, ptrain_data, layers)
        npt.assert_almost_equal(ecm, ecm_by_hand)
        tm.model_cascade(ecs_item, ptrain_data, layers, False)

    with pytest.raises(ValueError):
        tm.calc_layer_current(ecs_item, ptrain_data, ['unknown'])
    with pytest.raises(ValueError):
        tm.model_cascade(ecs_item, ptrain_data, ['unknown'], False)


def test_Nanduri2012():
    tsample = 0.01 / 1000
    tm = retina.Nanduri2012(tsample=tsample)

    # Assume 4 electrodes, each getting some stimulation
    pts = [stimuli.PulseTrain(tsample=tsample, dur=0.1)] * 4
    ptrain_data = [pt.data for pt in pts]

    # For each of these 4 electrodes, we have two effective current values:
    # one for the ganglion cell layer, one for the bipolar cell layer
    ecs_item = np.random.rand(2, 4)

    # ...and that should be the same as `calc_layer_current`:
    ecm_by_hand = np.sum(ecs_item[1, :, np.newaxis] * ptrain_data, axis=0)
    ecm = tm.calc_layer_current(ecs_item, ptrain_data)
    npt.assert_almost_equal(ecm, ecm_by_hand)

    # Running the model cascade:
    with pytest.raises(ValueError):
        tm.model_cascade(ecs_item, ptrain_data, ['GCL', 'INL'], False)
    with pytest.raises(ValueError):
        tm.model_cascade(ecs_item, ptrain_data, ['unknown'], False)

    # Regression test: Make sure Cython implementation reproduces legacy code
    tsample = 0.005 / 1000
    layers = ['GCL']
    use_jit = True
    stim = stimuli.PulseTrain(tsample, freq=20, amp=150, pulse_dur=0.45 / 1000,
                              dur=0.5)
    ecm = np.array([1, 1]).reshape((2, 1))
    legacy = LegacyNanduri2012(tsample=tsample)
    legacy_out = legacy.model_cascade(ecm, [stim.data], layers, use_jit)
    nanduri = retina.Nanduri2012(tsample=tsample)
    nanduri_out = nanduri.model_cascade(ecm, [stim.data], layers, use_jit)
    npt.assert_almost_equal(nanduri_out.data, legacy_out.data, decimal=2)
    npt.assert_almost_equal(nanduri_out.data[-1], legacy_out.data[-1],
                            decimal=2)


def test_Horsager2009_model_cascade():
    tsample = 0.01 / 1000
    tm = retina.Horsager2009(tsample=tsample)

    # Assume 4 electrodes, each getting some stimulation
    pts = [stimuli.PulseTrain(tsample=tsample, dur=0.1)] * 4
    ptrain_data = [pt.data for pt in pts]

    # For each of these 4 electrodes, we have two effective current values:
    # one for the ganglion cell layer, one for the bipolar cell layer
    ecs_item = np.random.rand(2, 4)

    # Run the model cascade:
    with pytest.raises(ValueError):
        tm.model_cascade(ecs_item, ptrain_data, ['GCL', 'INL'], False)
    with pytest.raises(ValueError):
        tm.model_cascade(ecs_item, ptrain_data, ['unknown'], False)
    tm.model_cascade(ecs_item, ptrain_data, ['GCL'], False)


def test_Horsager2009_calc_layer_current():
    tsample = 0.01 / 1000
    tm = retina.Horsager2009(tsample=tsample)

    # Assume 4 electrodes, each getting some stimulation
    pts = [stimuli.PulseTrain(tsample=tsample, dur=0.1)] * 4
    ptrain_data = [pt.data for pt in pts]

    # For each of these 4 electrodes, we have two effective current values:
    # one for the ganglion cell layer, one for the bipolar cell layer
    ecs_item = np.random.rand(2, 4)

    # Calulating layer current:
    # The Horsager model does not support INL, so it's just one layer:
    with pytest.raises(ValueError):
        tm.calc_layer_current(ecs_item, ptrain_data, ['GCL', 'INL'])
    with pytest.raises(ValueError):
        tm.calc_layer_current(ecs_item, ptrain_data, ['unknown'])

    # ...and that should be the same as `calc_layer_current`:
    ecm_by_hand = np.sum(ecs_item[1, :, np.newaxis] * ptrain_data, axis=0)
    ecm = tm.calc_layer_current(ecs_item, ptrain_data, ['GCL'])
    npt.assert_almost_equal(ecm, ecm_by_hand)


def test_Horsager2009():
    """Make sure the model can reproduce data from Horsager et al. (2009)

    Single-pulse data is taken from Fig.3, where the threshold current is
    reported for a given pulse duration and amplitude of a single pulse.
    We don't fit every data point to save some computation time, but make sure
    the model produces roughly the same values as reported in the paper for
    some data points.
    """

    def forward_pass(model, pdurs, amps):
        """Calculate model output based on a list of pulse durs and amps"""
        pdurs = np.array([pdurs]).ravel()
        amps = np.array([amps]).ravel()
        for pdur, amp in zip(pdurs, amps):
            in_arr = np.ones((2, 1))
            pt = stimuli.PulseTrain(model.tsample, amp=amp, freq=0.1,
                                    pulsetype='cathodicfirst',
                                    pulse_dur=pdur / 1000.0,
                                    interphase_dur=pdur / 1000.0)
            percept = model.model_cascade(in_arr, [pt.data], 'GCL', False)
            yield percept.data.max()

    def calc_error_amp(amp_pred, pdur, model):
        """Calculates the error in threshold current

        For a given data `pdur`, what is the `amp` needed for output `theta`?
        We're trying to find the current that produces output `theta`. Thus
        we calculate the error between output produced with `amp_pred` and
        `theta`.
        """
        theta_pred = list(forward_pass(model, pdur, amp_pred))[0]
        return np.log(np.maximum(1e-10, (theta_pred - model.theta) ** 2))

    def yield_fits(model, pdurs, amps):
        """Yields a threshold current by fitting the model to the data"""
        for pdur, amp in zip(pdurs, amps):
            yield scpo.fmin(calc_error_amp, amp, disp=0, args=(pdur, model))[0]

    # Data from Fig.3 in Horsager et al. (2009)
    pdurs = [0.07335, 0.21985, 0.52707, 3.96939]  # pulse duration in ms
    amps_true = [181.6, 64.7, 33.1, 14.7]  # threshold current in uA
    amps_expected = [201.9, 61.0, 29.2, 10.4]

    # Make sure our implementation comes close to ground-truth `amps`:
    # - Do the forward pass
    model = retina.Horsager2009(tsample=0.01 / 1000, tau1=0.42 / 1000,
                                tau2=45.25 / 1000, tau3=26.25 / 1000,
                                beta=3.43, epsilon=2.25, theta=110.3)
    amps_predicted = np.array(list(yield_fits(model, pdurs, amps_true)))

    # - Make sure predicted values still the same
    npt.assert_almost_equal(amps_expected, amps_predicted, decimal=0)


def test_Retina_Electrodes():
    logging.getLogger(__name__).info("test_Retina_Electrodes()")
    ssample = 1
    x_range = (-2, 2)
    y_range = (-3, 3)
    ret = retina.Grid(x_range=x_range, y_range=y_range, sampling=ssample,
                      save_data=False)
    npt.assert_equal(ret.gridx.shape, (int(np.diff(y_range) / ssample + 1),
                                       int(np.diff(x_range) / ssample + 1)))
    npt.assert_equal(ret.x_range, x_range)
    npt.assert_equal(ret.y_range, y_range)

    electrode1 = implants.Electrode('epiretinal', 1, 0, 0, 0)

    # Calculate current spread for all retinal layers
    retinal_layers = ['INL', 'OFL']
    cs = dict()
    ecs = dict()
    for layer in retinal_layers:
        cs[layer] = electrode1.current_spread(ret.gridx, ret.gridy,
                                              layer=layer)
        ecs[layer] = ret.current2effectivecurrent(cs[layer])

    electrode_array = implants.ElectrodeArray('epiretinal', [1, 1], [0, 1],
                                              [0, 1], [0, 1])
    npt.assert_equal(electrode1.x_center,
                     electrode_array.electrodes[0].x_center)
    npt.assert_equal(electrode1.y_center,
                     electrode_array.electrodes[0].y_center)
    npt.assert_equal(electrode1.radius, electrode_array.electrodes[0].radius)
    ecs_list, cs_list = ret.electrode_ecs(electrode_array)

    # Make sure cs_list has an entry for every layer
    npt.assert_equal(cs_list.shape[-2], len(retinal_layers))
    npt.assert_equal(ecs_list.shape[-2], len(retinal_layers))

    # Make sure manually generated current spreads match object
    for i, e in enumerate(retinal_layers):
        # last index: electrode, second-to-last: layer
        npt.assert_equal(cs[e], cs_list[..., i, 0])


def test_ret2dva():
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(retina.ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(retina.ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision


def test_dva2ret():
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(retina.dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(retina.dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision


def test_jansonius2009():
    # With `rho` starting at 0, all axons should originate in the optic disc
    # center
    for phi0 in [-135.0, 66.0, 128.0]:
        for loc_od in [(15.0, 2.0), (-15.0, 2.0), (-4.2, -6.66)]:
            ax_pos = retina.jansonius2009(phi0, n_rho=100,
                                          rho_range=(0.0, 45.0),
                                          loc_od=loc_od)
            npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
            npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for sign in [-1.0, 1.0]:
        for phi0 in [110.0, 135.0, 160.0]:
            ax_pos = retina.jansonius2009(sign * phi0, n_rho=801,
                                          loc_od=(15, 2),
                                          rho_range=(0.0, 45.0))
            print(ax_pos[-1, :])
            npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            retina.jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            retina.jansonius2009(0.0, n_rho=n_rho)

    # `rho_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            retina.jansonius2009(0.0, rho_range=(lorho, 45.0))
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            retina.jansonius2009(0.0, rho_range=(45.0, hirho))

    # `eye` must be left or right
    for eye in ['L', 'r', 'left', 'right']:
        with pytest.raises(ValueError):
            retina.jansonius2009(0.0, eye=eye)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
    for eye in ['LE', 'RE']:
        for loc_od in [(15.5, 1.5), (7.0, 3.0), (-2.0, -2.0)]:
            single_fiber = retina.jansonius2009(0, n_rho=1, loc_od=loc_od,
                                                rho_range=(0, 0))
            npt.assert_equal(len(single_fiber), 1)
            npt.assert_almost_equal(single_fiber[0], loc_od)


def test_find_closest_axon():
    phi = np.linspace(-180.0, 180.0, 10)
    axon_bundles = utils.parfor(retina.jansonius2009, phi)
    for idx, ax in enumerate(axon_bundles):
        # Each axon bundle should be closest to itself
        closest = retina.find_closest_axon(ax[-1, :], axon_bundles)
        npt.assert_almost_equal(closest, ax[-1:0:-1, :])


def test_axon_dist_from_soma():
    # A small grid
    xg, yg = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='xy')

    # When axon locations are snapped to the grid, a really short axon should
    # have zero distance to the soma:
    for x_soma in [-1.0, -0.2, 0.51]:
        axon = np.array([[i, i] for i in np.linspace(x_soma, x_soma + 0.01)])
        _, dist = retina.axon_dist_from_soma(axon, xg, yg)
        npt.assert_almost_equal(dist, 0.0)

    # On this simple grid, a diagonal axon should have dist [0, sqrt(2), 2]:
    for sign in [-1.0, 1.0]:
        for num in [10, 20, 50]:
            axon = np.array([[i, i] for i in np.linspace(sign, -sign, num)])
            _, dist = retina.axon_dist_from_soma(axon, xg, yg)
            npt.assert_almost_equal(dist, np.array([0.0, np.sqrt(2), 2.0]))

    # An axon that does not live near the grid should return infinite distance
    axon = np.array([[i, i] for i in np.linspace(1000.0, 1500.0)])
    _, dist = retina.axon_dist_from_soma(axon, xg, yg)
    npt.assert_equal(np.isinf(dist), True)


def test_axon_contribution():
    # Invalid calls
    for lmbda in [-1, 0]:
        with pytest.raises(ValueError):
            retina.axon_contribution([0], [0], decay_const=lmbda)
    for p in [-1, 0]:
        with pytest.raises(ValueError):
            retina.axon_contribution([0], [0], contribution_rule='mean',
                                     powermean_exp=p)
    with pytest.raises(ValueError):
        retina.axon_contribution([0], [0], contribution_rule='max',
                                 powermean_exp=1)
    with pytest.raises(ValueError):
        retina.axon_contribution([0], [0], sensitivity_rule='unknown')
    with pytest.raises(ValueError):
        retina.axon_contribution([0], [0], contribution_rule='unknown')

    # Make sure numbers are right for a simple setup:
    dist = np.arange(10)
    dist2 = (dist, dist)
    decay_const = 4.0
    for c_rule in ['max', 'mean', 'sum']:
        if c_rule == 'mean':
            powermean_exp = 1.0
        else:
            powermean_exp = None

        # If current spread == 1 everywhere, contribution given by `dist`
        _, contrib = retina.axon_contribution(
            dist2, np.ones_like(dist), sensitivity_rule='decay',
            decay_const=decay_const, contribution_rule=c_rule,
            powermean_exp=powermean_exp
        )
        sensitivity = np.exp(-dist / decay_const)
        method_to_call = getattr(np, c_rule)
        npt.assert_almost_equal(contrib, method_to_call(sensitivity))

    # No current spread in -> no axon contribution out
    for s_rule in ['decay', 'Jeng2011']:
        for c_rule in ['max', 'sum', 'mean']:
            for p in [1.0, 2.0, 3.0]:
                if c_rule == 'mean':
                    powermean_exp = p
                else:
                    powermean_exp = None
                _, contrib = retina.axon_contribution(
                    dist2, np.zeros_like(dist), sensitivity_rule=s_rule,
                    contribution_rule=c_rule, powermean_exp=powermean_exp
                )
                npt.assert_almost_equal(contrib, 0.0)


# deprecated
def test_make_axon_map():
    jan_x, jan_y = retina.jansonius(num_cells=10, num_samples=100)
    xg, yg = np.meshgrid(np.linspace(-100, 100, 21),
                         np.linspace(-100, 100, 21), indexing='xy')
    ax_id, ax_wt = retina.make_axon_map(xg, yg, jan_x, jan_y)

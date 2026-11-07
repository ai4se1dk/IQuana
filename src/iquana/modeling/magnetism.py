"""The magnetic parameters sasmodels adds to every compiled model with an SLD.

For each ``sld``-typed kernel parameter sasmodels appends a magnetisation
vector (``<sld>_M0``, ``<sld>_mtheta``, ``<sld>_mphi``) to the model's *call*
parameters, plus four polarisation parameters shared by the whole model
(``up_frac_i``, ``up_frac_f``, ``up_theta``, ``up_phi``). They are not in
``kernel_parameters``, so IQuana derives them here.

**Convention** (``sasmodels.details.convert_magnetism`` and ``kernel_iq.c``):
z is the beam axis and x, y are the detector axes. The magnetisation is
``M0 · (sin θ cos φ, sin θ sin φ, cos θ)`` with θ = ``mtheta`` measured from the
beam and φ = ``mphi`` from +x; the polarisation axis uses the same angles
(``up_theta``, ``up_phi``). sasmodels' own ``magnetism.rst`` writes **M**
differently; the formula above is what the code computes.

Magnetism is evaluated in the 2D kernel only. For 1D data sasmodels ignores
every parameter here.
"""

from typing import Any

import numpy as np

POLARIZATION_PARAMS = ('up_frac_i', 'up_frac_f', 'up_theta', 'up_phi')

#: Bounds for the magnetic SLD magnitudes, in 1e-6/Å². sasmodels declares them
#: unbounded; the generic fallback would give [-1, 1], below real materials
#: (iron is about 5).
M0_BOUNDS = (-10.0, 10.0)

#: Spin state of the beam before and after the sample for each preset, as
#: (incoming '+'?, outgoing: '+' / '-' / None for no analyser).
_BEAMS = {
    'unpolarised': (None, None),
    'sanspol+': ('+', None),
    'sanspol-': ('-', None),
    'pa++': ('+', '+'),
    'pa--': ('-', '-'),
    'pa+-': ('+', '-'),
    'pa-+': ('-', '+'),
}
BEAM_PRESETS = tuple(_BEAMS)


def unsupported_reason(kernel: Any) -> str | None:
    """Why *kernel* cannot be fitted with magnetism, or None when it can."""
    parameters = kernel.info.parameters
    # isinstance: mock kernels in tests expose arbitrary objects here.
    nmagnetic = getattr(parameters, 'nmagnetic', 0)
    if not isinstance(nmagnetic, int) or nmagnetic == 0:
        return (
            'the model has no magnetic SLD parameters (it has no SLD, or it is a '
            'pure-Python model, for which sasmodels generates none)'
        )
    if any(p.type == 'sld' and p.length > 1 for p in parameters.kernel_parameters):
        return (
            'the model has vector SLD parameters (such as sld[n]), which IQuana '
            'does not support yet'
        )
    return None


def derive_magnetic_params(kernel: Any) -> dict[str, dict[str, Any]]:
    """The magnetic call parameters of *kernel*, as IQuana parameter entries.

    Empty when the model does not support magnetism.
    """
    if unsupported_reason(kernel) is not None:
        return {}
    params: dict[str, dict[str, Any]] = {}
    for p in kernel.info.parameters.call_parameters:
        if p.type != 'magnetic':
            continue
        low, high = M0_BOUNDS if p.name.endswith('_M0') else p.limits
        params[p.name] = {
            'value': float(p.default),
            'min': float(low),
            'max': float(high),
            'vary': False,
            'description': p.description,
        }
    return params


def beam_fractions(
    beam: str, efficiency_in: float = 1.0, efficiency_out: float = 1.0
) -> tuple[float, float]:
    """``(up_frac_i, up_frac_f)`` for a beam preset.

    Fractions below 0.5 select the flipped spin state; an efficiency of 0.95 on
    a '+' state gives 0.95, on a '-' state 0.05. Without an analyser the
    outgoing fraction is 0.5.
    """
    if beam not in _BEAMS:
        raise ValueError(f'Unknown beam {beam!r}. Use one of: {", ".join(BEAM_PRESETS)}.')
    for name, value in (('efficiency_in', efficiency_in), ('efficiency_out', efficiency_out)):
        if not 0.5 <= value <= 1.0:
            raise ValueError(f'{name} must be between 0.5 and 1, got {value!r}.')
    incoming, outgoing = _BEAMS[beam]

    def fraction(state: str | None, efficiency: float) -> float:
        if state is None:
            return 0.5
        return efficiency if state == '+' else 1.0 - efficiency

    return fraction(incoming, efficiency_in), fraction(outgoing, efficiency_out)


def channel_factor(up_frac_i: float, up_frac_f: float) -> float:
    """Weight of the nuclear intensity in sasmodels' magnetic kernel.

    ``((1-ui)(1-uf) + ui·uf) / max(uf, 1-uf)``, from ``set_spin_weights`` in
    ``kernel_iq.c``. When every magnetic amplitude is zero sasmodels switches
    to the non-magnetic kernel, whose nuclear weight is 1. The two agree only
    where this factor is 1; elsewhere the model jumps as the magnetisation
    reaches zero.
    """
    ui = float(np.clip(up_frac_i, 0.0, 1.0))
    uf = float(np.clip(up_frac_f, 0.0, 1.0))
    return ((1 - ui) * (1 - uf) + ui * uf) / max(uf, 1 - uf)

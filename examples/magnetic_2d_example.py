"""
Example: 2D detector data and magnetic (polarised) SANS

sasmodels evaluates magnetic scattering for 2D data only, so the two features
come together. This script demonstrates how to:
1. Load a measured 2D image, mask the beam stop and the unusable pixels, fit it
2. Simulate a polarised (SANSPOL) image with known magnetisation
3. Enable magnetism with a beam preset and recover the truth
4. Meet the check that refuses a spin-flip fit whose magnetisation can vanish
5. Fit the measured SANSPOL+ and SANSPOL- images of magnetic nanoparticles
6. Save the analysis, mask included, and load it back with its result

Run from the repository root:  python examples/magnetic_2d_example.py
"""

import os
import tempfile
import warnings

from iquana import IQuana, examples, set_verbosity

# The package picks ASCII spellings for these when the console cannot encode
# them, which is what keeps this script runnable on a legacy Windows terminal.
from iquana.console import CHI_SQUARED, OK


def banner(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def free_values(result):
    return ', '.join(
        f'{name} = {info["value"]:.4g}'
        for name, info in result['parameters'].items()
        if not info['fixed']
    )


set_verbosity('quiet')

# ============================================================================
# Part 1: a measured 2D image
# ============================================================================
banner('Part 1: measured 2D data (P123 micelles)')

micelles = IQuana()
micelles.load_data(examples.example_path('p123_2d'))
print(f'2D data loaded: {micelles.is_2d}, {micelles.get_mask()["pixels"]} pixels')

# 33 pixels have zero intensity and zero dI, which the bumps engine cannot
# weight. Nothing is masked automatically: state it.
micelles.mask_ring(qmax=0.006)
micelles.mask_invalid()
mask = micelles.get_mask()
print(f'After masking: {mask["fitted"]} pixels fitted, {mask["user"]} excluded by masks')

micelles.set_model('sphere')
micelles.set_structure_factor('hardsphere')
micelles.set_resolution('none')
micelles.set_param('sld', value=0.4)
micelles.set_param('sld_solvent', value=6.4)
micelles.set_param('scale', value=0.05, min=1e-6, max=10, vary=True)
micelles.set_param('background', value=0.5, min=0, max=10, vary=True)
micelles.set_param('radius', value=50, min=10, max=200, vary=True)
micelles.set_param('radius_effective', value=60, min=10, max=300, vary=True)
micelles.set_param('volfraction', value=0.2, min=0, max=0.6, vary=True)

result = micelles.fit()
print(f'Fitted: {free_values(result)}')
print(f'Reduced {CHI_SQUARED}: {result["reduced_chisq"]:.3f}')
micelles.plot_results().show()

# ============================================================================
# Part 2: a simulated polarised image with known truth
# ============================================================================
banner('Part 2: simulated SANSPOL data')

TRUTH = {
    'radius': 60.0,
    'scale': 0.05,
    'background': 0.01,
    'sld': 2.0,
    'sld_solvent': 6.0,
    'sld_M0': 1.5,  # magnetic SLD, 1e-6/A^2
    'sld_mtheta': 90.0,  # magnetisation in the detector plane, along +x
}
image = examples.simulate_2d('sphere', qmax=0.15, npixels=64, beam='sanspol+', **TRUTH)
print(f'Simulated {image.data.size} pixels with sld_M0 = {image.truth["sld_M0"]}')

# ============================================================================
# Part 3: enable magnetism and recover the truth
# ============================================================================
banner('Part 3: magnetic fit')

fitter = IQuana()
fitter.set_data(image)
fitter.set_model('sphere')
fitter.set_resolution('none')
print(f'Magnetic parameters: {", ".join(fitter.get_magnetic_parameters())}')
fitter.enable_magnetism(beam='sanspol+')  # up_frac_i = 1, up_frac_f = 0.5
for name in ('sld', 'sld_solvent', 'background', 'sld_mtheta'):
    fitter.set_param(name, value=TRUTH[name])
fitter.set_param('radius', value=45, min=10, max=200, vary=True)
fitter.set_param('scale', value=0.03, min=1e-4, max=1, vary=True)
fitter.set_param('sld_M0', value=0.5, min=0.05, max=5, vary=True)

result = fitter.fit()
print(f'Fitted: {free_values(result)}')
print('Truth:  radius = 60, scale = 0.05, sld_M0 = 1.5')
fitter.plot_results().show()

# ============================================================================
# Part 4: the zero-magnetisation check
# ============================================================================
banner('Part 4: a spin-flip channel whose magnetisation can vanish')

spin_flip = IQuana()
spin_flip.set_data(image)
spin_flip.set_model('sphere')
spin_flip.enable_magnetism(beam='pa+-')
spin_flip.set_param('sld_mtheta', value=90)
spin_flip.set_param('sld_M0', value=1.0, min=0.0, max=5.0, vary=True)
try:
    spin_flip.fit()
except ValueError as error:
    print(f'Refused, as it should be:\n  {error}')

# ============================================================================
# Part 5: measured SANSPOL data
# ============================================================================
banner('Part 5: measured SANSPOL+ and SANSPOL- images')

for name in ('sanspol_plus', 'sanspol_minus'):
    channel = examples.load_fitter(name)  # mask, beam preset, starting values
    channel.set_resolution('none')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = channel.fit()
    print(f'{name:14s} reduced {CHI_SQUARED} {result["reduced_chisq"]:.3f}: {free_values(result)}')

# ============================================================================
# Part 6: save and reload
# ============================================================================
banner('Part 6: save the analysis and load it back')

path = os.path.join(tempfile.mkdtemp(), 'magnetic_analysis.json')
fitter.mask_ring(qmax=0.01)
fitter.fit()
fitter.save_analysis(path)
# The image was simulated in memory, so it is supplied again on reload.
restored = IQuana.load_analysis(path, data=image)
print(f'Magnetism enabled: {restored.is_magnetism_enabled()}')
print(f'Pixels masked: {restored.get_mask()["user"]}')
print(f'Fit result restored: {restored.fit_result is not None}')

print(f'\n{OK} 2D and magnetic example completed successfully!')

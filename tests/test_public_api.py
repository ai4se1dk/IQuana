"""Locks in the public API surface preserved across the 0.3 restructuring.

The package layout may move (data/, modeling/, inversion/), but everything
asserted here is a stability guarantee for downstream users: the names in
``__all__`` and the ``data_ops``/``pr_inversion`` module aliases must keep
working from the package root.
"""

import unittest

import iquana


class TestPublicApi(unittest.TestCase):
    def test_all_names_are_importable(self):
        for name in iquana.__all__:
            self.assertTrue(hasattr(iquana, name), f'missing public name: {name}')

    def test_expected_names_in_all(self):
        expected = {
            'IQuana',
            'MultiFitter',
            'DatasetHandle',
            'MultiFitResult',
            'MultiFitReport',
            'ConstraintError',
            'ExpressionError',
            'ParameterManager',
            'PD_DEFAULTS',
            'PD_DISTRIBUTION_TYPES',
            'get_all_models',
            'set_verbosity',
            'FitResultContract',
            'PosteriorSummary',
            'data_ops',
            'examples',
            'pr_inversion',
            'PrResult',
            'InsufficientDataError',
            'PrEstimationError',
            'GenericScatteringCalculator',
            'structure',
        }
        self.assertLessEqual(expected, set(iquana.__all__))

    def test_module_aliases_resolve_to_canonical_modules(self):
        from iquana import data_ops, pr_inversion

        self.assertEqual(data_ops.__name__, 'iquana.data.ops')
        self.assertEqual(pr_inversion.__name__, 'iquana.inversion')

    def test_structure_module_re_exports_sinccity(self):
        import sinccity

        self.assertEqual(iquana.structure.__name__, 'iquana.structure')
        self.assertIs(
            iquana.structure.GenericScatteringCalculator, sinccity.GenericScatteringCalculator
        )

    def test_weighting_api(self):
        fitter = iquana.IQuana()
        self.assertTrue(callable(fitter.set_weighting))
        self.assertEqual(fitter.get_weighting(), {'mode': 'dI'})

        from iquana.data import (  # noqa: F401
            WEIGHTING_MODES,
            WeightingSetting,
            apply_weighting,
            validate_weighting,
        )

        self.assertEqual(WEIGHTING_MODES, ('dI', 'sqrt', 'abs', 'none'))

    def test_pr_inversion_alias_exposes_public_entry_points(self):
        from iquana import pr_inversion

        for name in ('invert', 'auto_invert', 'estimate_alpha', 'estimate_n_terms', 'explore_dmax'):
            self.assertTrue(callable(getattr(pr_inversion, name)), name)

    def test_multifitter_exposes_its_documented_operations(self):
        from iquana import MultiFitter

        for name in (
            'add',
            'add_fitter',
            'remove',
            'share',
            'unshare',
            'link_params',
            'unlink_params',
            'constrain',
            'unconstrain',
            'set_param',
            'set_dataset_weight',
            'get_constraints',
            'get_sharing',
            'get_parameter_table',
            'describe',
            'calculate',
            'plot_model',
            'fit',
            'get_fit_report',
            'plot_results',
            'save_results',
        ):
            self.assertTrue(callable(getattr(MultiFitter, name)), name)

    def test_a_dataset_handle_does_not_offer_analysis_wide_operations(self):
        """A handle configures one dataset; it must not be able to fit it alone."""
        from iquana import DatasetHandle

        for name in ('fit', 'fit_bayesian', 'save_analysis', 'load_analysis', 'report'):
            self.assertFalse(hasattr(DatasetHandle, name), name)

    def test_constraint_errors_are_value_errors(self):
        """Existing ``except ValueError`` handlers keep working."""
        from iquana import ConstraintError, ExpressionError

        self.assertTrue(issubclass(ConstraintError, ValueError))
        self.assertTrue(issubclass(ExpressionError, ConstraintError))


if __name__ == '__main__':
    unittest.main()

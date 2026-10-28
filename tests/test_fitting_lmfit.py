import os
import unittest

from iquana import IQuana
from tests.helpers import create_decay_data_file


class TestLMFitFitting(unittest.TestCase):
    """Test LMFit fitting functionality."""

    def setUp(self):
        self.fitter = IQuana()
        self.data_file = create_decay_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_lmfit_fit_runs(self):
        try:
            result = self.fitter.fit(engine='lmfit', method='leastsq')

            self.assertIsNotNone(result)
            self.assertIn('engine', result)
            self.assertEqual(result['engine'], 'lmfit')
            self.assertIn('chisq', result)
            self.assertIn('parameters', result)
        except ValueError as error:
            if 'lmfit is not installed' in str(error):
                self.skipTest('lmfit not installed')
            raise

    def test_lmfit_fit_returns_parameters(self):
        try:
            result = self.fitter.fit(engine='lmfit', method='leastsq')

            self.assertIn('radius', result['parameters'])
            self.assertIn('scale', result['parameters'])

            for _param_name, param_result in result['parameters'].items():
                self.assertIn('value', param_result)
                self.assertIn('stderr', param_result)
                self.assertIn('formatted', param_result)
        except ValueError as error:
            if 'lmfit is not installed' in str(error):
                self.skipTest('lmfit not installed')
            raise

    def test_default_method_respects_bounds(self):
        # The data wants a radius far below min=10; the default must not go there.
        result = self.fitter.fit(engine='lmfit')

        self.assertEqual(result['method'], 'least_squares')
        self.assertGreaterEqual(result['parameters']['radius']['value'], 10.0)

    def test_leastsq_warns_when_leaving_bounds(self):
        with self.assertWarnsRegex(UserWarning, r'outside their bounds: radius'):
            result = self.fitter.fit(engine='lmfit', method='leastsq')

        self.assertLess(result['parameters']['radius']['value'], 10.0)

    def test_start_value_outside_bounds_is_refused_by_name(self):
        # Bounds moved past the value: least_squares cannot start there, and
        # scipy's own error does not say which parameter is wrong.
        self.fitter.set_param('radius', min=100.0, max=200.0)
        self.fitter.set_param('scale', value=5.0)
        for engine in ('lmfit', 'bumps'):
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(
                    ValueError,
                    r'outside its bounds: radius = 20 \(bounds \[100, 200\]\); '
                    r'scale = 5 \(bounds \[0\.01, 1\]\)',
                ):
                    self.fitter.fit(engine=engine)

    def test_varying_parameter_with_equal_bounds_is_refused_by_name(self):
        # min == max: least_squares refuses without a name, bumps crashes inside
        # its optimizer with "step has not accumulated enough history".
        self.fitter.set_param('radius', value=20.0, min=20.0, max=20.0)
        for engine in ('lmfit', 'bumps'):
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(
                    ValueError,
                    r'^Bounds leave no room to fit: radius has min = max = 20 '
                    r'\(hold it with vary=False instead\)\.$',
                ):
                    self.fitter.fit(engine=engine)

    def test_varying_parameter_with_crossed_bounds_is_refused_by_name(self):
        # The value test alone would say "move the value inside the bounds",
        # which no value can satisfy.
        self.fitter.set_param('radius', min=30.0, max=10.0)
        with self.assertRaisesRegex(
            ValueError,
            r'^Bounds leave no room to fit: radius has min 30 above max 10 '
            r'\(swap the bounds\)\.$',
        ):
            self.fitter.fit(engine='lmfit')

    def test_empty_range_and_bad_start_are_reported_together(self):
        self.fitter.set_param('radius', value=20.0, min=20.0, max=20.0)
        self.fitter.set_param('scale', value=5.0)
        with self.assertRaisesRegex(
            ValueError,
            r'radius has min = max = 20 .*\. Starting value outside its bounds: '
            r'scale = 5 \(bounds \[0\.01, 1\]\)',
        ):
            self.fitter.fit(engine='lmfit')

    def test_fixed_parameter_with_equal_bounds_does_not_block_the_fit(self):
        self.fitter.set_param('sld', value=2.0, min=2.0, max=2.0)
        result = self.fitter.fit(engine='lmfit')
        self.assertEqual(result['method'], 'least_squares')

    def test_fixed_parameter_outside_bounds_does_not_block_the_fit(self):
        self.fitter.set_param('sld', min=10.0, max=20.0)
        result = self.fitter.fit(engine='lmfit')
        self.assertEqual(result['method'], 'least_squares')

    def test_varying_pd_width_outside_unit_interval_is_refused(self):
        self.fitter.enable_polydispersity(True)
        self.fitter.set_pd_param('radius', pd_width=1.5, vary=True)
        with self.assertRaisesRegex(ValueError, r'radius_pd = 1\.5 \(bounds \[0, 1\]\)'):
            self.fitter.fit(engine='lmfit')

    def test_bayesian_sampling_checks_start_values_too(self):
        self.fitter.set_param('radius', min=100.0, max=200.0)
        with self.assertRaisesRegex(ValueError, r'outside its bounds: radius = 20'):
            self.fitter.fit_bayesian(samples=10, burn=1)


class TestLMFitFittingWithPolydispersity(unittest.TestCase):
    """Test scipy/lmfit fitting engine with polydispersity parameters."""

    def setUp(self):
        self.fitter = IQuana()
        self.data_file = create_decay_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_fit_with_pd_enabled_fixed_width_leastsq(self):
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=False)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='lmfit', method='leastsq')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('parameters', result)

    def test_fit_with_pd_enabled_varying_width_least_squares(self):
        self.fitter.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='lmfit', method='least_squares')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('radius_pd', result['parameters'])

    def test_pd_width_updated_after_lmfit(self):
        self.fitter.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        self.fitter.enable_polydispersity(True)

        self.fitter.fit(engine='lmfit', method='least_squares')

        pd_config = self.fitter.get_pd_param('radius')
        self.assertIsNotNone(pd_config['pd'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

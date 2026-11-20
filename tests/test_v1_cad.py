"""Optional geometry-only checks; no mesh, solver, or extraction."""
import importlib.util
import unittest

from emulateqpu.chain import linear_chain_geometry


@unittest.skipUnless(importlib.util.find_spec('shapely'), 'Optional physical geometry dependencies unavailable')
class LinearChainCAD(unittest.TestCase):
    def test_six_data_eleven_modes_fortyfour_checked_terminals(self):
        request, conductors = linear_chain_geometry(6)
        self.assertEqual(request['geometry']['n_qubits'], 11)
        self.assertEqual(len(request['mode_order']), 11)
        self.assertEqual(len(request['terminal_mapping']), 11)
        self.assertEqual(len(conductors), 44)
        self.assertEqual(len(request['preview']['links']), 10)
        self.assertGreater(request['preview']['min_conductor_gap_um'], 0.)
        terminal_names = []
        for mapping in request['terminal_mapping'].values():
            terminal_names += mapping['islands']+[mapping['drive'], mapping['readout']]
        self.assertEqual(set(terminal_names), set(conductors))
        self.assertTrue(all(shape.is_valid and shape.geom_type == 'Polygon' for shape in conductors.values()))

    def test_legacy_three_and_five_requests_and_polygons_identical(self):
        from emulateqpu._physics import chain_family as frozen
        for n in (3, 5):
            expected = frozen.chain_request(n)
            polygons, _ = frozen.preview_polygons(expected['geometry'], expected['family'])
            actual, shapes = linear_chain_geometry(n)
            self.assertEqual(actual, expected)
            self.assertEqual(set(shapes), set(polygons))
            for key in shapes:
                self.assertEqual(shapes[key].wkb, polygons[key].wkb)

    def test_resource_limit_and_invalid_spacing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'max_modes'):
            linear_chain_geometry(6, max_modes=9)
        with self.assertRaises(ValueError):
            linear_chain_geometry(6, spacing_um=0.)
        with self.assertRaises(ValueError):
            linear_chain_geometry(6, spacing_um=100.)


if __name__ == '__main__':
    unittest.main()

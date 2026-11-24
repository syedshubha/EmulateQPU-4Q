"""Geometry and model checks for the four-qubit junction layout."""
import importlib.util
import unittest

from emulateqpu.fullwave.design import four_qubit_design
from emulateqpu.fullwave.junction_layout import (
    COUPLER_Y_MM,
    CRITICAL_CURRENT_DENSITY_A_M2,
    SQUID_BRANCH_OFFSET_MM,
    add_junction_layout,
)
from emulateqpu.fullwave.layout import DATABASE_GRID_MM


@unittest.skipUnless(importlib.util.find_spec("shapely"), "Physical profile required")
class JunctionLayoutTests(unittest.TestCase):
    def _inputs(self, topology="dedicated"):
        design, sheets = four_qubit_design(readout_topology=topology)
        conductors = {name: sheets[name] for name in design["conductors"]}
        return design, conductors

    def test_both_readout_topologies_have_ten_isolated_tunnel_overlaps(self):
        for topology in ("dedicated", "shared"):
            with self.subTest(topology=topology):
                design, conductors = self._inputs(topology)
                before = conductors.copy()
                sheets, devices, loops = add_junction_layout(design, conductors)
                self.assertEqual(len(sheets), 33)
                self.assertEqual(len(devices), 10)
                self.assertEqual(len(loops), 3)
                self.assertEqual(
                    {name for name, shape in conductors.items() if shape is not before[name]},
                    set(design["island_order"]),
                )
                for island in design["island_order"]:
                    self.assertTrue(conductors[island].covers(before[island]))
                self.assertFalse(any(name.startswith("port_jj_") for name in sheets))
                for device in devices:
                    bottom = sheets[device["bottom_electrode"]]
                    top = sheets[device["top_electrode"]]
                    barrier = sheets[device["barrier"]]
                    left, right = (conductors[name] for name in device["islands"])
                    self.assertTrue(bottom.intersection(top).equals(barrier))
                    self.assertGreater(bottom.intersection(left).area, 0)
                    self.assertGreater(top.intersection(right).area, 0)
                    self.assertTrue(bottom.disjoint(right))
                    self.assertTrue(top.disjoint(left))
                    for x, y in barrier.exterior.coords:
                        self.assertAlmostEqual(x / DATABASE_GRID_MM, round(x / DATABASE_GRID_MM), places=7)
                        self.assertAlmostEqual(y / DATABASE_GRID_MM, round(y / DATABASE_GRID_MM), places=7)
                    expected_current = barrier.area * 1e-6 * CRITICAL_CURRENT_DENSITY_A_M2
                    self.assertAlmostEqual(device["critical_current_A"], expected_current, places=17)
                    self.assertLess(abs(device["critical_current_A"] / device["target_critical_current_A"] - 1), .011)

    def test_matched_squids_keep_nominal_idle_energy_and_enclose_flux_apertures(self):
        from shapely.geometry import Point

        design, conductors = self._inputs()
        sheets, devices, loops = add_junction_layout(design, conductors)
        by_label = {device["label"]: device for device in devices}
        by_mode = {row["label"]: row for row in design["junctions"]}
        self.assertEqual([loop["coupler"] for loop in loops], ["c01", "c12", "c23"])
        self.assertEqual([device["label"] for device in devices if device["center_mm"][1] == 0],
                         ["q0", "q1", "q2", "q3"])
        for loop in loops:
            aperture = sheets[loop["aperture"]]
            self.assertTrue(aperture.contains(Point(*loop["center_mm"])))
            self.assertGreater(aperture.area, 0)
            self.assertEqual(loop["normal"], "+Z")
            lower, upper = [by_label[name] for name in loop["junctions"]]
            self.assertAlmostEqual(lower["center_mm"][1], COUPLER_Y_MM - SQUID_BRANCH_OFFSET_MM)
            self.assertAlmostEqual(upper["center_mm"][1], COUPLER_Y_MM + SQUID_BRANCH_OFFSET_MM)
            self.assertEqual(lower["junction_energy_GHz"], upper["junction_energy_GHz"])
            self.assertEqual(lower["islands"], upper["islands"])
            target = by_mode[loop["coupler"]]["EJ_GHz"]
            self.assertEqual(loop["effective_idle_EJ_GHz"], target)
            self.assertLess(abs(loop["estimated_effective_idle_EJ_GHz"] / target - 1), .011)

    def test_unintended_electrode_contact_leaves_input_unchanged(self):
        from shapely.geometry import box

        design, conductors = self._inputs()
        conductors["obstruction"] = box(2.99995, -0.00005, 3.00005, .00005)
        before = conductors.copy()
        with self.assertRaisesRegex(ValueError, "Junction electrode.*incorrect contacts"):
            add_junction_layout(design, conductors)
        self.assertEqual(set(conductors), set(before))
        self.assertTrue(all(shape is before[name] for name, shape in conductors.items()))

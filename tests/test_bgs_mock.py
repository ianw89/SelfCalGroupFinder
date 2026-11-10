import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))
from groupcatalog import BGSMockGroupCatalog


class BGSMockGroupCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "clustering.fits")
        self.table = Table({
            'TARGETID': np.array([2**53 + 1, 2**53 + 2, 3, 4, 5], dtype=np.int64),
            'RA': [10., 11., 12., 13., 14.],
            'DEC': [1., 2., 3., 4., 5.],
            'Z': [0.1, 0.2, 0.1, 0.6, 0.1],
            'R_MAG_APP': [19.0, 19.52, 19.52, 18.0, np.nan],
            'R_MAG_ABS': [-20., -21., -21., -22., -20.],
            'G_R_REST': [0.5, 0.9, 0.8, 0.8, 0.8],
            'G_R_OBS': [0.7, 1.0, 0.9, 0.9, 0.9],
            'PHOTSYS': ['S', 'N', 'S', 'S', 'S'],
            'WEIGHT': [1., 2., 3., 4., 5.],
        })
        self.table.write(self.path, format="fits")
        self.catalog = BGSMockGroupCatalog(self.path, mag_cut=19.5, frac_area=0.1)
        self.catalog.set_output_folder(self.tmp.name + "/")

    def test_preprocess_and_postprocess_alignment(self):
        self.catalog.preprocess()
        main = np.loadtxt(self.catalog.preprocess_file, ndmin=2)
        sidecar = pd.read_pickle(self.catalog.file_pattern + "_galprops.pkl")
        self.assertEqual(main.shape, (2, 6))
        np.testing.assert_array_equal(sidecar['TARGETID'], [2**53 + 1, 2**53 + 2])
        np.testing.assert_allclose(main[:, 0], [10., 11.])
        np.testing.assert_allclose(main[:, 2], [0.1, 0.2])
        np.testing.assert_allclose(main[:, 4], main[:, 4].clip(min=1))
        np.testing.assert_array_equal(main[:, 5], sidecar['QUIESCENT'].astype(int))
        np.testing.assert_array_equal(sidecar['WEIGHT'], [1., 2.])
        np.testing.assert_array_equal(sidecar['Z_ASSIGNED_FLAG'], [0, 0])
        self.assertEqual(self.catalog.GF_props['zmin'], 0.001)
        self.assertEqual(self.catalog.GF_props['zmax'], 0.5)

        output = np.column_stack((main[:, :3], 10 ** main[:, 3], main[:, 4],
                                  [0., 0.], [1e12, 1e12], [1, 1],
                                  [1e10, 1e10], [0, 0], [1., 1.], [1., 1.]))
        np.savetxt(self.catalog.GF_outfile, output)
        self.catalog.postprocess()
        np.testing.assert_array_equal(self.catalog.all_data['TARGETID'], sidecar['TARGETID'])
        self.assertEqual(len(self.catalog.centrals), 1)
        self.assertEqual(len(self.catalog.sats), 1)

        np.savetxt(self.catalog.GF_outfile, output[:1])
        with self.assertRaisesRegex(ValueError, "different|rows"):
            self.catalog.postprocess()

    def test_missing_columns_and_bad_configuration_fail(self):
        self.table.remove_column('G_R_REST')
        self.table.write(self.path, format="fits", overwrite=True)
        with self.assertRaisesRegex(ValueError, 'G_R_REST'):
            self.catalog.preprocess()
        with self.assertRaisesRegex(ValueError, 'frac_area'):
            BGSMockGroupCatalog(self.path, 19.5, 0)
        with self.assertRaisesRegex(ValueError, 'fluxlim'):
            BGSMockGroupCatalog(self.path, 19.5, 0.1, gfprops={'color': 1})
        with self.assertRaisesRegex(ValueError, 'latent'):
            BGSMockGroupCatalog(self.path, 19.5, 0.1,
                                gfprops={'fluxlim': 1, 'latent': np.eye(4)})

    def test_duplicate_ids_rejected(self):
        self.table['TARGETID'][1] = self.table['TARGETID'][0]
        self.table.write(self.path, format="fits", overwrite=True)
        with self.assertRaisesRegex(ValueError, 'not unique'):
            self.catalog.preprocess()


if __name__ == '__main__':
    unittest.main()

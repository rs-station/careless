from careless.io.asu import ReciprocalASU,ReciprocalASUCollection
import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi


@pytest.mark.parametrize('anomalous', [True, False])
@pytest.mark.parametrize('dmin', [10., 5.])
def test_reciprocal_asu(dmin, anomalous, cell_and_spacegroups):
    for cell,sg in cell_and_spacegroups:
        rasu = ReciprocalASU(cell, sg, dmin, anomalous)
        Hall = rs.utils.generate_reciprocal_asu(
            cell,
            sg,
            dmin,
            anomalous
        )

        centric = rs.utils.is_centric(Hall, sg)
        assert np.all(rasu.centric == centric)

        multiplicity = rs.utils.compute_structurefactor_multiplicity(Hall, sg)
        assert np.all(rasu.multiplicity == multiplicity)

        d_hkl = cell.calculate_d_array(rasu.lookup_table.get_hkls())
        assert np.all(d_hkl >= dmin)

        refl_id = rasu.to_refl_id(Hall)
        assert np.all(refl_id == np.arange(len(Hall)))

        miller_index = rasu.to_miller_index(np.arange(len(Hall)))
        assert np.all(miller_index == Hall)


@pytest.mark.parametrize('anomalous', [[True, True], [True, False], [False, False]])
@pytest.mark.parametrize('dmin', [[10., 10.], [5., 10.], [5., 5.]])
def test_double_reciprocal_asu_collection(dmin, anomalous, cell_and_spacegroups):
    for cell,sg in cell_and_spacegroups:
        rasus = [ReciprocalASU(cell, sg, d, a) for d,a in zip(dmin,anomalous)]
        rac = ReciprocalASUCollection(rasus)

        Hall_per_asu = [rs.utils.generate_reciprocal_asu(cell,sg,d,a) for d,a in zip(dmin,anomalous)]
        n = np.sum([len(i) for i in Hall_per_asu])

        assert len(rac.lookup_table) == n
        refl_ids = []
        centrics = []
        multiplicities = []
        for asu_id,h in enumerate(Hall_per_asu):
            refl_id = rac.to_refl_id(asu_id*np.ones((len(h), 1)), h)
            asu_id_test,h_test = rac.to_asu_id_and_miller_index(refl_id)
            assert np.all(asu_id == asu_id_test)
            assert np.all(h == h_test)
            refl_ids.append(refl_id.flatten())
            centrics.append(rs.utils.is_centric(h, sg))
            multiplicities.append(rs.utils.compute_structurefactor_multiplicity(h, sg))

        refl_ids = np.concatenate(refl_ids)
        #There should be no duplicates
        assert len(refl_ids) == len(np.unique(refl_ids))

        #There should be no gaps
        assert np.all(refl_ids == np.arange(len(refl_ids)))

        centrics = np.concatenate(centrics)
        assert np.all(rac.centric == centrics)

        multiplicities = np.concatenate(multiplicities)
        assert np.all(rac.multiplicity == multiplicities)

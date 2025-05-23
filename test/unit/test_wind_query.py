import numpy as np
import numpy.random as rng
import ard.wind_query as wq
import floris

import pytest


class TestWindQuery:

    def setup_method(self):
        self.query = wq.WindQuery()

    def test_setters(self):

        # set up wind condition values
        size_q = (5,)
        dir_q = 360.0 * rng.rand(*size_q)
        V_q = np.abs(15.0 * rng.randn(*size_q))

        # set in the wind conditions to query
        self.query.set_directions(dir_q)
        self.query.set_speeds(V_q)

        # make sure values are actually set in exactly
        assert np.all(
            self.query.get_directions() == dir_q
        ), "specified directions should match"
        assert np.all(self.query.get_speeds() == V_q), "specified speeds should match"
        assert (
            self.query.N_conditions == size_q[0]
        ), "internal size tracking should match"
        assert self.query.is_valid()

    def test_resetters(self):

        # set up wind condition values
        size_q = (5,)
        dir_q = 360.0 * rng.rand(*size_q)
        V_q = np.abs(15.0 * rng.randn(*size_q))

        # set in the wind conditions to query
        self.query.set_directions(dir_q)
        self.query.set_speeds(V_q)

        # make sure values are actually set in exactly
        assert np.all(
            self.query.get_directions() == dir_q
        ), "specified directions should match"
        assert np.all(self.query.get_speeds() == V_q), "specified speeds should match"
        assert (
            self.query.N_conditions == size_q[0]
        ), "internal size tracking should match"
        assert self.query.is_valid()

        # now, let's modify wind condition values
        size_q = (4,)

        # start with the direction
        dir_q = 360.0 * rng.rand(*size_q)
        self.query.set_directions(dir_q)

        # now direction should have new values, but the query shouldn't be valid
        # and should therefore raise an error
        with pytest.raises(AssertionError):
            np.all(
                self.query.get_directions() == dir_q
            ), "specified directions should match"
        assert (
            self.query.N_conditions is None
        ), "number of conditions should be ill-defined"
        assert self.query.is_valid() is False, "not valid with different lengths"

        # now modify speeds
        V_q = np.abs(15.0 * rng.randn(*size_q))
        self.query.set_speeds(V_q)

        # make sure values are actually set in exactly and the query should be valid
        assert np.all(self.query.get_speeds() == V_q), "specified speeds should match"
        assert np.all(
            self.query.get_directions() == dir_q
        ), "specified directions should match"
        assert (
            self.query.N_conditions == size_q[0]
        ), "internal size tracking should match"
        assert self.query.is_valid()

    def test_winddata(self):
        wind_directions = np.array([250, 260, 270])
        wind_speeds = np.array([5, 6, 7, 8, 9, 10])
        ti_table = 0.06

        # generate a wind rose
        wr = floris.WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            ti_table=ti_table,
        )
        # meshgrid out into a single data stream
        WS, WD = [V.flatten() for V in np.meshgrid(wind_speeds, wind_directions)]

        # override query, building from the FLORIS data obj
        self.query = wq.WindQuery.from_FLORIS_WindData(wr)

        # make sure the result is legit
        assert np.all(
            np.equal(self.query.get_directions(), WD.flatten())
        ), "specified directions should match"
        assert np.all(
            np.equal(self.query.get_speeds(), WS.flatten())
        ), "specified speeds should match"
        assert (
            self.query.N_conditions == wind_directions.size * wind_speeds.size
        ), "internal size tracking should match"
        assert self.query.is_valid()

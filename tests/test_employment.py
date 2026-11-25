import numpy as np
import pytest

from mixuse.config import ConfigError, merge_params
from mixuse.employment import context_rates, scenario_jobs
from mixuse.params import defaults


def test_local_employment_uses_only_same_function_and_excludes_territory():
    areas = np.array([10_000, 4_000, 6_000, 100_000, 12_000, 20_000])
    jobs = np.zeros((6, 3, 2))
    jobs[0, 0] = (900, 900)  # территория
    jobs[1, 0] = (40, 20)
    jobs[2, 1] = (30, 10)
    jobs[4, 0] = (48, 24)
    jobs[5, 2] = (60, 30)
    rates = context_rates(areas, jobs, [0], [1, 2, 3])
    assert rates['К'] == pytest.approx((.007, .0035, 1, 1))
    assert rates['ОД'] == pytest.approx((.005, 10 / 6000, 1, 0))
    assert rates['П'] == pytest.approx((.003, .0015, 0, 1))


def test_floor_capacity_and_lower_bound_are_separate_from_context():
    params = defaults()
    rates = {'К': (.01, .0025, 3, 1), 'ОД': (.002, 0, 2, 0), 'П': (0, 0, 0, 0)}
    shares = {'К': .25, 'ОД': .25, 'П': .5}
    jobs, known, capacity = scenario_jobs(10_000, .14, shares, params, rates)
    assert jobs == pytest.approx(12.5 + 5)
    assert known == pytest.approx(12.5 / 4)
    assert capacity == pytest.approx(1400 * (.25 / 28 + .25 / 15 + .5 / 69))
    assert scenario_jobs(10_000, .14, shares, params, None) == pytest.approx([capacity] * 3)


def test_jobs_method_validation():
    assert merge_params({})['jobs_method'] == 'local'
    assert merge_params({'jobs_method': 'legacy'})['jobs_method'] == 'legacy'
    with pytest.raises(ConfigError, match='jobs_method'):
        merge_params({'jobs_method': 'unknown'})

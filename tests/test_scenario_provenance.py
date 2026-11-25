"""Регрессия происхождения сценариев и сохранения исходных долей."""
import json

import pandas as pd
import pytest

from mixuse.checks import builtin_sensitivity, estimates_check, sensitivity, threshold_check
from mixuse.compare import compare
from mixuse.export import blocks_geojson, scenario_map_name, table_csv
from mixuse.employment import JOB_GROUPS
from mixuse.params import (SCENARIOS, EMPIRICAL_SCENARIOS,
                           EXPERIMENTAL_SCENARIOS, paired_scenarios, SCENARIO_ORIGINS)
from mixuse.scenarios import validate_scenarios
from test_rules_v5 import v5  # noqa: F401 — fixture


def test_all_shares_sum_to_100():
    for dataset in (SCENARIOS, EMPIRICAL_SCENARIOS, EXPERIMENTAL_SCENARIOS, paired_scenarios()):
        assert all(sum(s.values()) == 100 for s in dataset.values())


def test_experimental_proportions_preserved():
    assert EXPERIMENTAL_SCENARIOS['К+ОД+П'] == {'Ж': 0, 'К': 10, 'ОД': 40, 'П': 50}
    assert EXPERIMENTAL_SCENARIOS['Ж+К+П'] == {'Ж': 50, 'К': 10, 'ОД': 0, 'П': 40}
    assert EXPERIMENTAL_SCENARIOS['Ж+К+ОД'] == {'Ж': 75, 'К': 5, 'ОД': 20, 'П': 0}


def test_paired_unique_names_and_provenance():
    paired = paired_scenarios()
    assert len(paired) == 7
    assert set(paired) == set(SCENARIO_ORIGINS)
    assert not validate_scenarios(paired)


def test_canonical_names_match_nonzero_functions():
    for dataset in (SCENARIOS, EMPIRICAL_SCENARIOS, EXPERIMENTAL_SCENARIOS):
        assert not validate_scenarios(dataset)
        for name, shares in dataset.items():
            assert name == '+'.join(group for group, share in shares.items() if share > 0)

    # Старые пользовательские подписи допускаются ради совместимости.
    assert not validate_scenarios({'ОД+П': {'Ж': 0, 'К': 10, 'ОД': 40, 'П': 50}})


def test_paired_tables_keep_all_results(v5):
    cfg, layers, city = v5
    territory, params = [21], cfg['params']

    def tables(scenarios):
        full, _, _ = compare(city, territory, params, scenarios)
        return (
            (full, ('',), 'Как есть'),
            (sensitivity(city, territory, params, scenarios),
             ('изменение K ', 'школы: без свободного места '), 'K окружения как есть'),
            (estimates_check(city, territory, params, scenarios), ('K окружения ',), 'K окружения Как'),
            (builtin_sensitivity(layers, cfg, params, territory, scenarios), ('K окружения ',), 'K окружения Как'),
            (threshold_check(layers, cfg, params, territory, scenarios), ('K окружения ',), 'K окружения Как'),
        )

    paired = paired_scenarios()
    paired_tables = tables(paired)
    assert list(paired_tables[0][0].columns) == ['Как есть', *paired]
    for source, suffix, origin in ((EMPIRICAL_SCENARIOS, 'эмп.', 'эмпирический'),
                                   (EXPERIMENTAL_SCENARIOS, 'эксп.', 'экспериментальный')):
        for (joint, prefixes, baseline), (separate, _, _) in zip(paired_tables, tables(source), strict=True):
            assert joint.columns.is_unique
            pd.testing.assert_series_equal(joint[baseline], separate[baseline])
            for name, shares in source.items():
                label = f'{name} [{suffix}]'
                assert paired[label] == shares
                assert SCENARIO_ORIGINS[label] == origin
                for prefix in prefixes:
                    pd.testing.assert_series_equal(joint[prefix + label], separate[prefix + name], check_names=False)

    full = paired_tables[0][0]
    for name in ('Ж+К+ОД', 'К+ОД'):
        capacity = 'Вместимость проектной площади, рабочих мест'
        assert full.at[capacity, f'{name} [эмп.]'] != full.at[capacity, f'{name} [эксп.]']


def test_paired_exports_keep_full_names(v5, tmp_path):
    cfg, _, city = v5
    paired = paired_scenarios()
    full, _, detail = compare(city, [21], cfg['params'], paired)
    names = ['Как есть', *paired]
    assert list(detail['short'].columns) == names
    assert list(detail['conclusions']) == list(paired)
    assert list(detail['effect']) == names

    for label, table in (('full', full), ('short', detail['short'])):
        csv_path = tmp_path / f'{label}.csv'
        table_csv(table, csv_path)
        assert list(pd.read_csv(csv_path, index_col=0).columns) == names

    geojson_path = tmp_path / 'blocks.geojson'
    blocks_geojson(city, geojson_path, detail, paired)
    features = json.loads(geojson_path.read_text(encoding='utf-8'))['features']
    for k, feature in zip(city['table'].index, features, strict=True):
        access = {key: value for key, value in feature['properties'].items() if key.startswith('доступ ')}
        assert access == {f'доступ {name}': detail['effect'][name].get(k, 0) for name in paired}

    assert {scenario_map_name(name, paired) for name in paired} == {f'карта_{name}.png' for name in paired}


@pytest.mark.parametrize('method', ['local', 'legacy'])
def test_layer_distributes_scenario_jobs_without_changing_outer_blocks(v5, tmp_path, method):
    cfg, _, city = v5
    table = city['table'].copy()
    table.loc[table.block_id == 20, 'area'] *= 2
    city = dict(city, table=table)
    name = 'К+ОД+П'
    scenarios = {name: SCENARIOS[name]}
    _, _, detail = compare(city, [20, 21], dict(cfg['params'], jobs_method=method), scenarios)
    path = tmp_path / 'blocks.geojson'
    blocks_geojson(city, path, detail)
    layer = json.loads(path.read_text(encoding='utf-8'))
    assert layer['метод расчёта рабочих мест'] == method
    result = detail['values'][name]
    fields = [f'занятость_{group} {name}' for group in JOB_GROUPS]
    inside = []
    for feature in layer['features']:
        p = feature['properties']
        assert p[f'рабочие места {name}'] == pytest.approx(sum(p[key] for key in fields) + p[f'занятость_прочие {name}'])
        if p['территория']:
            inside.append(p)
        else:
            assert p[f'рабочие места {name}'] == p['jobs']
            assert p[f'изменение рабочих мест {name}'] == 0
    assert sum(p[f'рабочие места {name}'] for p in inside) == pytest.approx(result['J'])
    assert sum(p[f'изменение рабочих мест {name}'] for p in inside) == pytest.approx(result['dJ'])
    assert inside[0][f'рабочие места {name}'] == pytest.approx(2 * inside[1][f'рабочие места {name}'])
    for group, key in zip(JOB_GROUPS, fields, strict=True):
        assert sum(p[key] for p in inside) == pytest.approx(result['jobs_by_function'][group])


@pytest.mark.parametrize('name', ['Как есть', '_source', '_warnings'])
def test_scenario_names_cannot_overwrite_reserved_results(v5, name):
    cfg, _, city = v5
    scenarios = {name: SCENARIOS['Ж+К']}
    assert validate_scenarios(scenarios)
    with pytest.raises(ValueError, match='Зарезервированные названия вариантов'):
        compare(city, [21], cfg['params'], scenarios)

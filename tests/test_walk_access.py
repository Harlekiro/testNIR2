"""Пешеходный доступ к К и ОД на искусственном примере и сохранённых матрицах."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from mixuse import export
from mixuse.blocks import build_table, load_layers
from mixuse.compare import compare
from mixuse.config import load_config
from mixuse.params import paired_scenarios
from mixuse.provision_bn import file_sha256, save_matrix
from mixuse.walk_access import WalkAccessError, load_walk_access, walk_neighbors
from test_page import NODE, run_page
from test_rules_v5 import v5  # noqa: F401 — fixture


def test_reachability_and_alternatives(v5):
    cfg, _, city = v5
    city['table'] = city['table'].copy()
    table = city['table']
    table['есть_К'] = False
    table['есть_ОД'] = table.block_id.isin([21, 23, 24])
    params = dict(cfg['params'], r=300)
    scenario = {'Ж+К': {'Ж': 50, 'К': 50, 'ОД': 0, 'П': 0}}
    access = {'method': 'walk', 'graph': 'walk', 'units': 'minutes', 'block_ids': [20, 21, 22, 23, 24],
              'rows': [[], [], [1, 12, 2, 0, 3, 10, 4, 9], [], []],
              'threshold_minutes': 10, 'max_minutes': 20}

    _, _, geometric = compare(city, [21], params, scenario)
    _, _, network = compare(city, [21], params, scenario, walk_access=access)
    assert geometric['values']['Ж+К']['acc']['К'] == (100, 0)
    assert network['values']['Ж+К']['acc']['К'] == (0, 0)

    full, info, network = compare(city, [21, 24], params, scenario, walk_access=access)
    assert '10 мин' in info['доступ к К и ОД']
    assert full.at['Способ расчёта доступа к К и ОД', 'Ж+К'] == info['доступ к К и ОД']
    assert network['values']['Ж+К']['acc']['К'] == (100, 0)
    assert network['values']['Ж+К']['acc']['ОД'] == (0, 0)  # сохраняется квартал 23, ровно 10 минут
    _, _, shorter = compare(city, [21, 24], params, scenario, walk_access=dict(access, threshold_minutes=9))
    assert shorter['values']['Ж+К']['acc']['ОД'] == (0, 100)
    assert shorter['effect_by_function']['Ж+К'] == {'К': {2: 1}, 'ОД': {2: -1}}
    assert shorter['effect']['Ж+К'][2] == 0  # суммарная окраска не маскирует раздельные слои
    with pytest.raises(WalkAccessError, match='максимум'):
        walk_neighbors(dict(access, threshold_minutes=21), table.block_id, [2])


def test_matrix_validation_clipping_and_missing_fallback(v5, tmp_path):
    cfg, _, city = v5
    cfg['_path'] = str(tmp_path / 'configs' / 'v5.yaml')
    path = tmp_path / 'page.json'
    export.web_json_v5(city, cfg, path, prefer_walk=True)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['access_method'] == 'geometry' and 'walk_access' not in data
    assert 'геометрическое расстояние' in data['walk_access_note']

    ids = list(city['table'].block_id)
    values = np.full((len(ids), len(ids)), np.inf, dtype=np.float16)
    np.fill_diagonal(values, 0)
    values[2, [0, 1, 3, 4]] = [8, 10.0078125, 15, 21]
    metadata = {'граф': 'walk', 'единицы': 'минуты', 'sha256 слоя кварталов': file_sha256(cfg['blocks'])}
    matrix = pd.DataFrame(values, index=ids, columns=ids)
    save_matrix(cfg, matrix, metadata)
    access, _ = load_walk_access(cfg, ids, 10.005, maximum=20)
    assert walk_neighbors(access, ids, [2])[2].tolist() == [0, 2]
    assert walk_neighbors(access, ids, [0])[0].tolist() == [0]  # направление важно

    # При обрезке остаются 20, 21 и 22, а не исходный индекc 3 для квартала 23.
    export.web_json_v5(city, cfg, path, clip=400, prefer_walk=True)
    data = json.loads(path.read_text(encoding='utf-8'))
    access = data['walk_access']
    assert access['block_ids'] == [20, 21, 22] == [b['id'] for b in data['blocks']]
    assert access['rows'][2] == [0, 8, 1, 10.0078125, 2, 0]
    assert data['access_method'] == 'walk' and access['source'] == 'OpenStreetMap'
    assert load_walk_access(cfg, list(reversed(ids)))[0] is None
    save_matrix(cfg, matrix, dict(metadata, **{'sha256 слоя кварталов': 'invalid'}))
    assert load_walk_access(cfg, ids)[0] is None


def test_notebook_exports_walk_mode(tmp_path, monkeypatch):
    import test_notebook

    cells = test_notebook.code_cells()
    monkeypatch.setattr(test_notebook, 'code_cells', lambda: [s.replace('ПЕШКОМ = False', 'ПЕШКОМ = True') for s in cells])
    ns = test_notebook.run_notebook('vysotsk', str(tmp_path))
    data = json.loads((tmp_path / 'vysotsk_для_страницы.json').read_text(encoding='utf-8'))
    assert data['access_method'] == 'walk'
    assert data['walk_access']['threshold_minutes'] == 10
    assert 'пешком ≤ 10 мин' in ns['info']['доступ к К и ОД']


def test_existing_walk_matrix_is_reused_for_school_preparation(v5, monkeypatch):
    from mixuse import bn_bridge, provision_bn, provision_estimate

    cfg, _, city = v5
    monkeypatch.setattr(provision_estimate, 'load_page_data', lambda *_: None)
    monkeypatch.setattr(provision_bn, 'load_matrix', lambda *_, **__: (object(), {}))
    monkeypatch.setattr(bn_bridge, 'blocksnet_python', lambda: 'python')
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bn_bridge.subprocess, 'run', run)
    bn_bridge.prepare(cfg, city)
    assert [cmd[3] for cmd in commands] == ['provision']


@pytest.mark.parametrize('code', ['vysotsk', 'gatchina_layers'])
def test_walk_page_matches_python(code, tmp_path):
    if not NODE:
        pytest.skip('Node.js не найден')
    cfg = load_config(Path(ROOT) / 'configs' / f'{code}.yaml')
    city = build_table(load_layers(cfg, cfg['_groups']), cfg, cfg['params'], cfg['_groups'])
    cfg['scenarios'] = paired_scenarios()
    path = tmp_path / 'page.json'
    export.web_json_v5(city, cfg, path, prefer_walk=True)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert 'walk_access' in data and len(data['scenarios']) == 7
    params, territory = cfg['params'], cfg['territory']
    kit = {'mode': 'own', 'value': params['kit']} if params.get('kit') is not None else {
        'mode': 'sp', 'spRow': params['kit_sp_row']}
    jobs = [{'territory': territory, 'kit': kit, 'params': {'walk_minutes': threshold}}
            for threshold in (5, 10, 20)]
    pages = run_page(path, jobs, tmp_path)
    _, _, geometric = compare(city, territory, params, cfg['scenarios'])
    for threshold, page in zip((5, 10, 20), pages, strict=True):
        walk = dict(data['walk_access'], threshold_minutes=threshold)
        _, info, detail = compare(city, territory, params, cfg['scenarios'], walk_access=walk)
        assert page['accessLabel'] == info['доступ к К и ОД']
        assert list(page['out']) == ['Как есть', *cfg['scenarios']]
        assert page['effByFunction'] == {n: {g: {str(i): v for i, v in group.items()}
                                            for g, group in change.items()}
                                         for n, change in detail['effect_by_function'].items()}
        for name, calc in detail['values'].items():
            old, js = geometric['values'][name], page['out'][name]
            for key in ('N', 'J', 'K', 'sP', 'kP', 'sT', 'kT', 'sNo', 'kNo'):
                assert calc[key] == old[key]
                assert js[key] == pytest.approx(calc[key], rel=1e-3, abs=0.01)
            for group, val in calc['acc'].items():
                assert js['acc'][group] == pytest.approx(val, abs=0.05)
            assert calc['acc'].get('П') == old['acc'].get('П')

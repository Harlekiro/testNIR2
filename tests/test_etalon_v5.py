"""Эталоны правил v5 и исторические сценарии ноутбука Высоцка и Гатчины.

Эталоны посчитаны командой `python -m mixuse compare configs/<код>.yaml --out tests/etalon/v5/<код>` (19.11.2026).
Петербург: проверяются таблица кварталов, сравнение, шаг 7 и 4.8.3–4.8.4; шаг 7б и 4.8.5 (по 3–4 пересчёта
таблицы, около 3 минут) — только для Высоцка и Гатчины. Если данных набора нет — тест пропускается.
"""
import os

os.environ['MIXUSE_PROVISION'] = 'fallback'   # эталоны — с запасным расчётом школ и садов (4.7.3)

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, ETALON_V5, HISTORICAL_SCENARIOS, same_strings
from test_notebook import run_notebook
from mixuse.config import load_config
from mixuse.blocks import load_layers, build_table
from mixuse.compare import compare
from mixuse.checks import sensitivity, builtin_sensitivity, estimates_check, stats_check, threshold_check

ETALON = ETALON_V5
SETS = ['vysotsk', 'gatchina_layers', 'spb_full']
SLOW = {'spb_full'}
pytestmark = pytest.mark.filterwarnings('ignore:FigureCanvasAgg is non-interactive')


def ref(code, name, **kw):
    return pd.read_csv(os.path.join(ETALON, code, f'{code}_{name}.csv'), encoding='utf-8-sig', **kw)


def notebook_reference_path(code, filename):
    legacy = os.path.join(ETALON, '..', 'v5_legacy', code, filename)
    if os.path.exists(legacy):
        return legacy
    return os.path.join(ETALON, code, filename)


STR = dict(dtype=str, keep_default_na=False)


def same_table(a, b):
    """Таблица кварталов: числа — с точностью округления 0,1, остальное — как строки."""
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c]):
            assert np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)).max() <= 0.1 + 1e-9, c
        else:
            norm = lambda s: s.astype(object).where(s.notna(), '').astype(str).to_numpy()
            assert (norm(a[c]) == norm(b[c])).all(), c


@pytest.fixture(scope='module', params=SETS)
def run(request):
    code = request.param
    cfg = load_config(os.path.join(ROOT, 'configs', f'{code}.yaml'))
    if not os.path.exists(cfg['blocks']):
        pytest.skip(f'нет данных набора {code}')
    if code == 'spb_full':
        # Эталоны Петербурга зафиксированы до обновления долей по умолчанию.
        cfg['scenarios'] = HISTORICAL_SCENARIOS
    L = load_layers(cfg, cfg['_groups'])
    city = build_table(L, cfg, cfg['params'], cfg['_groups'])
    return code, cfg, L, city


def test_blocks_table(run):
    code, cfg, L, city = run
    got = city['table'].drop(columns=['cx', 'cy']).round(1).reset_index(drop=True)
    same_table(ref(code, 'таблица_кварталов'), got)


def test_comparison_short_and_detailed(run):
    code, cfg, L, city = run
    full, info, det = compare(city, cfg['territory'], cfg['params'], cfg['scenarios'])
    same_strings(ref(code, 'сравнение_вариантов', index_col=0, **STR), det['short'].astype(str))
    same_strings(ref(code, 'сравнение_подробно', index_col=0, **STR), full.astype(str))


def test_checks(run):
    code, cfg, L, city = run
    P, SCN, T = cfg['params'], cfg['scenarios'], cfg['territory']
    same_strings(ref(code, 'устойчивость', **STR), sensitivity(city, T, P, SCN).astype(str))
    same_strings(ref(code, 'оценки', **STR), estimates_check(city, T, P, SCN).astype(str))
    st = stats_check(city, P)
    assert (st is None) == (not os.path.exists(os.path.join(ETALON, code, f'{code}_статистика.csv')))
    if st is not None:
        same_strings(ref(code, 'статистика', **STR), st.astype(str))
    if code in SLOW:
        return
    same_strings(ref(code, 'встроенные', **STR), builtin_sensitivity(L, cfg, P, T, SCN).astype(str))
    same_strings(ref(code, 'порог_цеха', **STR), threshold_check(L, cfg, P, T, SCN).astype(str))


@pytest.mark.parametrize('code', ['vysotsk', 'gatchina_layers'])
def test_notebook_v5(code, tmp_path):
    """Ноутбук воспроизводит эталоны для своих явно заданных сценариев."""
    cfg = load_config(os.path.join(ROOT, 'configs', f'{code}.yaml'))
    if not os.path.exists(cfg['blocks']):
        pytest.skip(f'нет данных набора {code}')
    out = str(tmp_path)
    run_notebook(code, out)
    got = lambda name, **kw: pd.read_csv(os.path.join(out, f'{code}_{name}.csv'), encoding='utf-8-sig', **kw)
    same_table(ref(code, 'таблица_кварталов'), got('таблица_кварталов'))
    for name, kw in [('сравнение_вариантов', {'index_col': 0}), ('сравнение_подробно', {'index_col': 0}),
                     ('устойчивость', {}), ('встроенные', {}), ('оценки', {}), ('порог_цеха', {}), ('здания', {})]:
        path = notebook_reference_path(code, f'{code}_{name}.csv')
        same_strings(pd.read_csv(path, encoding='utf-8-sig', **kw, **STR), got(name, **kw, **STR))

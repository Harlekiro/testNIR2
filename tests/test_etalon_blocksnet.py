"""Эталон с точным расчётом школ и садов через blocksnet (tests/etalon/v5_blocksnet/, раздел 12).

Эталоны tests/etalon/v5/ посчитаны запасным расчётом 4.7.3, чтобы тесты не зависели от blocksnet. Здесь — те же
сравнения, но строки 12.5 посчитаны blocksnet по всему набору (без ускорения 12.4), и обеспеченность «как есть».
Эталоны посчитаны 20.11.2026 в окружении с blocksnet (Python 3.11); Петербург — около 18 мин:
    python -m mixuse compare configs/<код>.yaml --out tests/etalon/v5_blocksnet/<код> --fast
    python -m mixuse provision configs/<код>.yaml --out tests/etalon/v5_blocksnet/<код>
(из папки оставлены только файлы сравнения и обеспеченности).
Если blocksnet не установлен или нет данных и матрицы набора — тесты пропускаются.
"""
import os

import pandas as pd
import pytest

from conftest import ROOT, ETALON_BLOCKSNET, HISTORICAL_SCENARIOS, same_strings
from mixuse.config import load_config
from mixuse.blocks import load_layers, build_table
from mixuse.compare import compare
from mixuse import provision_bn as pb

ETALON = ETALON_BLOCKSNET
SETS = ['vysotsk', 'gatchina_layers', 'spb_full']
STR = dict(dtype=str, keep_default_na=False)
pytestmark = pytest.mark.filterwarnings('ignore:FigureCanvasAgg is non-interactive')


def ref(code, name):
    return pd.read_csv(os.path.join(ETALON, code, f'{code}_{name}.csv'), encoding='utf-8-sig', index_col=0, **STR)


@pytest.fixture(scope='module', params=SETS)
def run(request):
    code = request.param
    try:
        pb.require_blocksnet()
    except pb.BlocksnetError:
        pytest.skip('blocksnet не установлен (запускайте в окружении с Python 3.11)')
    cfg = load_config(os.path.join(ROOT, 'configs', f'{code}.yaml'))
    if not os.path.exists(cfg['blocks']) or not os.path.exists(pb.matrix_paths(cfg)[0]):
        pytest.skip(f'нет данных или матрицы набора {code}')
    city = build_table(load_layers(cfg, cfg['_groups']), cfg, cfg['params'], cfg['_groups'])
    mx, _ = pb.load_matrix(cfg, city['table']['block_id'].tolist())
    return code, cfg, city, mx


def test_comparison_with_blocksnet(run):
    """Краткая и подробная таблицы со строками 12.5 из blocksnet совпадают с эталоном."""
    code, cfg, city, mx = run
    P, SCN, T = dict(cfg['params'], jobs_method='legacy'), HISTORICAL_SCENARIOS, cfg['territory']
    _, _, det = compare(city, T, P, SCN)
    E_ids = city['table']['block_id'].iloc[det['E']].astype(int).tolist()
    prov = pb.variant_provision(city, mx, T, E_ids, det['values'])
    full, info, det = compare(city, T, P, SCN, prov=prov)
    assert info['школы и сады'].startswith('blocksnet')
    same_strings(ref(code, 'сравнение_вариантов'), det['short'].astype(str))
    same_strings(ref(code, 'сравнение_подробно'), full.astype(str))


def test_as_is_with_blocksnet(run):
    """Обеспеченность «как есть» (команда provision, 12.8) совпадает с эталоном."""
    code, cfg, city, mx = run
    summary, _ = pb.as_is_report(cfg, city, mx, cfg['territory'])
    got = summary.astype(str)
    got.index = got.index.astype(str)
    same_strings(ref(code, 'обеспеченность'), got)

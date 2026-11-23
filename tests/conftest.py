"""Маленький искусственный пример: 4 квартала, числа проверяются вручную (см. test_small_example.py).

Кварталы 100 × 100 м (UTM 35N, EPSG:32635; координаты — от точки 500 000, 6 600 000):
    10: x 0–100,     y 0–100    — территория; склад 20 × 20 м, 2 этажа, сервис warehouse → П
    11: x 200–300,   y 0–100    — жилой дом 20 × 20 м, 5 этажей, внутри кафе (К) и банк (ОД);
                                   нежилое 15 × 15 м, 1 этаж, без сервисов, зона ТД1 → ОД
    12: x 0–100,     y 200–300  — жилой дом 10 × 10 м, 1 этаж, жилая площадь 50 м²;
                                   нежилое 10 × 10 м (меньше S_min — не учитывается); школа на 100 мест
    13: x 1000–1100, y 0–100    — нежилое 20 × 20 м, 1 этаж, зона ТР2 → X; детский сад без числа мест
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mixuse.geometry import utm_to_lonlat  # noqa: E402

X0, Y0 = 500000.0, 6600000.0
UTM = 'urn:ogc:def:crs:EPSG::32635'
WGS = 'urn:ogc:def:crs:OGC:1.3:CRS84'


def sq(x, y, w, h=None):
    h = w if h is None else h
    return [[[X0 + x, Y0 + y], [X0 + x + w, Y0 + y], [X0 + x + w, Y0 + y + h], [X0 + x, Y0 + y + h], [X0 + x, Y0 + y]]]


def pt(x, y):
    return [X0 + x, Y0 + y]


def _to_deg(coords):
    if isinstance(coords[0], (int, float)):
        return utm_to_lonlat(coords, 35).tolist()
    return [_to_deg(c) for c in coords]


def write_fc(path, feats, crs=UTM):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for gtype, coords, props in feats:
        if crs == WGS:
            coords = _to_deg(coords)
        out.append({'type': 'Feature', 'properties': props, 'geometry': {'type': gtype, 'coordinates': coords}})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'crs': {'type': 'name', 'properties': {'name': crs}}, 'features': out}, f, ensure_ascii=False)


def make_small(folder, other_crs=UTM, territory=(10,), extra=''):
    """Пишет слои маленького примера и файл настроек. other_crs — система зданий, сервисов и зон."""
    d = os.path.join(folder, 'data')
    write_fc(os.path.join(d, 'blocks.geojson'), [
        ('Polygon', sq(0, 0, 100), {'block_id': 10}), ('Polygon', sq(200, 0, 100), {'block_id': 11}),
        ('Polygon', sq(0, 200, 100), {'block_id': 12}), ('Polygon', sq(1000, 0, 100), {'block_id': 13})])
    write_fc(os.path.join(d, 'buildings.geojson'), [
        ('Polygon', sq(40, 40, 20), {'area': 400, 'living': 0, 'floors': 2, 'liv_area': 0}),
        ('Polygon', sq(210, 10, 20), {'area': 400, 'living': 1, 'floors': 5, 'liv_area': None}),
        ('Polygon', sq(250, 50, 15), {'area': 225, 'living': 0, 'floors': 1, 'liv_area': 0}),
        ('Polygon', sq(10, 210, 10), {'area': 100, 'living': 1, 'floors': 1, 'liv_area': 50}),
        ('Polygon', sq(50, 250, 10), {'area': 100, 'living': 0, 'floors': 1, 'liv_area': 0}),
        ('Polygon', sq(1040, 40, 20), {'area': 400, 'living': 0, 'floors': None, 'liv_area': 0})], other_crs)
    sv = os.path.join(d, 'services')
    write_fc(os.path.join(sv, 'warehouse.geojson'), [('Point', pt(50, 50), {})], other_crs)
    write_fc(os.path.join(sv, 'cafe.geojson'), [('Point', pt(215, 15), {})], other_crs)
    write_fc(os.path.join(sv, 'bank.geojson'), [('Point', pt(225, 25), {})], other_crs)
    write_fc(os.path.join(sv, 'school.geojson'), [('Point', pt(80, 280), {'capacity': 100})], other_crs)
    write_fc(os.path.join(sv, 'kindergarten.geojson'), [('Point', pt(1080, 80), {'capacity': None})], other_crs)
    write_fc(os.path.join(sv, 'park.geojson'), [('Point', pt(90, 90), {})], other_crs)          # группа «нет»
    write_fc(os.path.join(sv, 'outpost.geojson'), [('Point', pt(95, 95), {})], other_crs)       # нет в таблице групп
    write_fc(os.path.join(d, 'zones.geojson'), [
        ('Polygon', sq(-10, -10, 120), {'code': 'ТП1'}), ('Polygon', sq(190, -10, 120), {'code': 'ТД1'}),
        ('Polygon', sq(990, -10, 120), {'code': 'ТР2'})], other_crs)
    cfg = os.path.join(folder, 'small.yaml')
    kit = '' if 'params:' in extra else 'params: {kit: 1.4, kit_source: пример}\n'
    with open(cfg, 'w', encoding='utf-8') as f:
        f.write(f"""name: small
mode: layers
blocks: data/blocks.geojson
block_id_field: block_id
buildings: data/buildings.geojson
fields: {{footprint: area, is_living: living, storeys: floors, living_area: liv_area}}
services: data/services
zones: data/zones.geojson
zone_field: code
zone_rules: [[ТП, П], [ТД, ОД]]
territory: {list(territory)}
scenarios:
  "С1 Ж+ОД": {{Ж: 60, К: 10, ОД: 30, П: 0}}
  "С2 ОД+П": {{Ж: 0, К: 10, ОД: 40, П: 50}}
{kit}{extra}""")
    return cfg


def same_strings(ref, got):
    """Таблицы совпадают строка в строку (значения сравниваются как текст)."""
    assert list(ref.columns) == list(got.columns)
    assert list(map(str, ref.index)) == list(map(str, got.index))
    bad = ref.to_numpy() != got.to_numpy()
    assert not bad.any(), ref.where(bad).stack().to_dict()


ETALON_V5 = os.path.join(ROOT, 'tests', 'etalon', 'v5')                       # эталоны правил раздела 4 (запасной 4.7.3)
ETALON_BLOCKSNET = os.path.join(ROOT, 'tests', 'etalon', 'v5_blocksnet')      # то же с точным расчётом blocksnet

# Входные сценарии исторических эталонов Петербурга и точного BlocksNet.
HISTORICAL_SCENARIOS = {
    'Ж+К': {'Ж': 85, 'К': 15, 'ОД': 0, 'П': 0},
    'Ж+К+ОД': {'Ж': 75, 'К': 5, 'ОД': 20, 'П': 0},
    'К+ОД': {'Ж': 0, 'К': 15, 'ОД': 85, 'П': 0},
    'ОД+П': {'Ж': 0, 'К': 10, 'ОД': 40, 'П': 50},
    'Ж+П': {'Ж': 50, 'К': 10, 'ОД': 0, 'П': 40},
}


@pytest.fixture
def small_cfg(tmp_path):
    return make_small(str(tmp_path))


@pytest.fixture
def small_cfg_mixed(tmp_path):
    """Тот же пример, но здания, сервисы и зоны — в градусах (как у Высоцка)."""
    return make_small(str(tmp_path), other_crs=WGS)

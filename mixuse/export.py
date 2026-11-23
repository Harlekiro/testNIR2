"""Сохранение результатов: CSV, слой GeoJSON для QGIS, JSON для страницы, карта PNG, отчёт.

Имена файлов и столбцов — как в results/ ноутбука.
"""
import json
import os
import re

import numpy as np
import pandas as pd

from .walk_access import DEFAULT_PREPARED_MINUTES, DEFAULT_WALK_MINUTES, load_walk_access


def out_path(out_dir, code, suffix):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{code}_{suffix}')


def blocks_csv(city, path):
    """Таблица кварталов (results/<код>_таблица_кварталов.csv), округление до 0,1."""
    city['table'].drop(columns=['cx', 'cy']).round(1).to_csv(path, index=False, encoding='utf-8-sig')
    return path


def table_csv(df, path, index=True):
    df.to_csv(path, index=index, encoding='utf-8-sig')
    return path


def report_txt(lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def _geom(g):
    return {'type': 'MultiPolygon', 'coordinates': [[r.tolist() for r in p] for p in g[1]]}


def _blocks_crs(city):
    return 'urn:ogc:def:crs:OGC:1.3:CRS84' if city['epsg'] in (4326, None) else f'urn:ogc:def:crs:EPSG::{city["epsg"]}'


def blocks_geojson(city, path, det=None, SCN=None):
    """Слой кварталов для QGIS в исходной системе координат кварталов.
    Если передано сравнение — поля «территория», «окружение», «доступ <вариант>» (+1 появились, −1 пропали)."""
    inT = set(det['T']) if det else set(); inE = set(det['E'].tolist()) if det else set()
    feats = []
    for k, row in city['table'].iterrows():
        props = {c: (None if pd.isna(row[c]) else (bool(row[c]) if isinstance(row[c], (bool, np.bool_)) else (row[c] if isinstance(row[c], str) else float(row[c]))))
                 for c in ['block_id', 'zone', 'area', 'pop', 'jobs', 'есть_К', 'есть_ОД', 'есть_П', 'cap_school', 'cap_kinder']}
        if det:
            props['территория'] = k in inT; props['окружение'] = k in inE
            for s in SCN:
                props['доступ ' + s] = det['effect'][s].get(k, 0)
                for group, changes in det.get('effect_by_function', {}).get(s, {}).items():
                    props[f'изменение доступа {group} {s}'] = changes.get(k, 0)
        feats.append({'type': 'Feature', 'properties': props, 'geometry': _geom(city['orig'][k])})
    collection = {'type': 'FeatureCollection', 'crs': {'type': 'name', 'properties': {'name': _blocks_crs(city)}},
                  'features': feats}
    if det and 'access_method' in det:
        collection['способ расчёта доступа к К и ОД'] = det['access_method']
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(collection, f, ensure_ascii=False)
    return path


def scenario_map_name(name, scenarios):
    """Имя PNG с полным названием варианта или его прежним уникальным номером С1, С2, …"""
    numbered = re.match(r'^([СC]\d+)\s', name)
    if numbered:
        number = numbered[1]
        if not any(other != name and (other == number or re.match(rf'^{number}\s', other))
                   for other in scenarios):
            name = number
    safe_name = re.sub(r'[<>:"/\\|?*%\x00-\x1f]', lambda match: f'%{ord(match[0]):02X}', name)
    return f'карта_{safe_name}.png'


def map_png(city, det, show, R, title, path):
    """Шаг 8. Карта в файл PNG (командная строка: без окна на экране)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = map_figure(city, det, show, R, title)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def map_figure(city, det, show, R, title):
    """Шаг 8. Рисунок карты: территория — фиолетовая, окружение обведено; зелёный — функции появились,
    оранжевый — пропали. Возвращает рисунок matplotlib (ноутбук показывает его и сохраняет)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    eff = det['effect'][show]; inT = set(det['T']); inE = set(det['E'].tolist())
    verts, face, edge, lw = [], [], [], []
    for k, polys in enumerate(city['polys']):
        e = eff.get(k, 0)
        fc = '#4b3f8f' if k in inT else ('#3f9e5f' if e > 0 else '#f0a030' if e < 0 else ('#e8e4d8' if k in inE else '#f4f4f2'))
        for p in polys:
            verts.append(p[0]); face.append(fc); edge.append('#333333' if k in inE else '#bbbbbb'); lw.append(0.8 if k in inE else 0.3)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(PolyCollection(verts, facecolors=face, edgecolors=edge, linewidths=lw))
    allxy = np.vstack(verts); ax.set_xlim(allxy[:, 0].min(), allxy[:, 0].max()); ax.set_ylim(allxy[:, 1].min(), allxy[:, 1].max())
    ax.set_aspect('equal'); ax.axis('off')
    ax.legend(handles=[Patch(color='#4b3f8f', label='территория'), Patch(facecolor='#e8e4d8', edgecolor='#333333', label=f'окружение {R:g} м'),
                       Patch(color='#3f9e5f', label='функции появились'), Patch(color='#f0a030', label='функции пропали')],
              loc='lower left', fontsize=9, frameon=False)
    access = det.get('access_method', '')
    caption = f'\nДоступ к К и ОД: {access}' if access.startswith('OpenStreetMap') else ''
    ax.set_title(f'{title}: вариант {show}{caption}', fontsize=11)
    return fig


def buildings_csv(city, path):
    """Таблица зданий (results/<код>_здания.csv): шаг таблицы 4.4, категория и рабочие места каждого здания,
    координата точки на поверхности в системе координат кварталов — чтобы проверить правила на карте в QGIS
    («Добавить слой из текста с разделителями», поля x и y)."""
    d = city['detail']['buildings']
    t = city['table']
    proj = city['proj']
    x, y = d['ix'].to_numpy(), d['iy'].to_numpy()
    if proj:                                             # обратно в градусы
        x, y = x / proj.kx + proj.lon0, y / proj.ky + proj.lat0
    out = pd.DataFrame({
        'block_id': t['block_id'].to_numpy()[d['block'].to_numpy()],
        'x': np.round(x, 6 if proj else 1), 'y': np.round(y, 6 if proj else 1),
        'жилое': d['living'].astype(int),
        'этажность_не_указана': d['storeys_missing'].astype(int),
        'площадь_застройки': d['footprint'].round(1),
        'поэтажная_площадь': d['floor_area'].round(1),
        'шаг_4_4': d['step'],
        'категория': d['category'],
        'рабочих_мест_А': d['jobs_A'].round(2),
        'рабочих_мест_Б': d['jobs_B'].round(2),
        'встроенные_К': d['bi_К'].round(1), 'встроенные_ОД': d['bi_ОД'].round(1),
        'рабочих_мест_встроенных': d['jobs_builtin'].round(2),
        'жителей': d['pop'].round(1),
    })
    out.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def web_json_v5(city, cfg, path, pdat=None, clip=None, *, walk_minutes=DEFAULT_WALK_MINUTES,
                walk_max_minutes=None, prefer_walk=False):
    """Файл для страницы (этап 7): формат страницы страница/конструктор.html —
    кварталы с контурами в метрах и показателями, точки метро и остановок, параметры (КИТ: kit, kit_source, kit_sp_row),
    варианты и исходная геометрия для экспорта GeoJSON. Если есть точный расчёт blocksnet «как есть» (pdat),
    в файл входят данные для оценки обеспеченности (provision).
    clip — оставить кварталы, центр которых не дальше clip м от центра территории (для большого города;
    totals — по всему набору)."""
    from .groups import zone_group
    from .params import SP42_B1, SCENARIO_ORIGINS, paired_scenarios
    t = city['table']
    P, rules = cfg['params'], cfg.get('zone_rules', [])
    keep = np.arange(len(t))
    T_ids = [int(i) for i in cfg.get('territory') or []]
    if clip and T_ids:
        tt = t[t['block_id'].isin(T_ids)]
        cx, cy = tt['cx'].mean(), tt['cy'].mean()
        keep = np.where(np.hypot(t['cx'] - cx, t['cy'] - cy) <= clip)[0]
    new = {int(k): i for i, k in enumerate(keep)}
    blocks = []
    for k in keep:
        row = t.iloc[k]
        z = row['zone']
        zg = zone_group(z, rules, cfg.get('_zone_dict'))
        blocks.append({
            'id': int(row['block_id']), 'z': zg if zg in ('П', 'ОД') else '',
            'zc': '' if z is None or (isinstance(z, float) and np.isnan(z)) else str(z).replace('LandUse.', ''),
            'a': round(float(row['area']), 1), 'cx': round(float(row['cx']), 2), 'cy': round(float(row['cy']), 2),
            'pop': round(float(row['pop']), 2), 'j': round(float(row['jobs']), 2), 'jA': round(float(row['jobs_A']), 2),
            'K': int(row['есть_К']), 'OD': int(row['есть_ОД']), 'P': int(row['есть_П']),
            'sc': round(float(row['cap_school'])), 'kg': round(float(row['cap_kinder'])), 'fa': round(float(row['fa']), 1),
            'nb': int(row['n_bld']), 'nbd': int(row['n_bld_default']),
            'g': [[np.round(r, 2).tolist() for r in p] for p in city['polys'][k]],
            'geo': _geom(city['orig'][k])['coordinates'],
        })
    from .compare import CITY_SHARE
    params = {k: P.get(k) for k in ('kit', 'kit_source', 'kit_sp_row', 'R', 'r', 'q', 'k_liv', 'a_К', 'a_ОД', 'a_П',
                                    'n_school', 'n_kinder', 'housing_note')}
    params['city_share'] = CITY_SHARE        # порог сравнения с городом, % (18.2а): страница берёт его отсюда, своего числа нет
    web = {'format': 'konstruktor-smeshannosti', 'format_version': 2, 'rules': 'v5',
           'name': cfg.get('title') or cfg['name'], 'code': cfg['name'], 'territory': T_ids,
           'totals': {'pop': float(t['pop'].sum()), 'jobs': float(t['jobs'].sum()), 'jobsA': float(t['jobs_A'].sum())},
           'clip': float(clip or 0), 'params': params, 'scenarios': cfg['scenarios'],
           'paired_scenarios': paired_scenarios(), 'scenario_origins': SCENARIO_ORIGINS,
           'blocks_crs': _blocks_crs(city),
           'sp42': {k: [n, v] for k, (n, v) in SP42_B1.items()},
           'blocks': blocks,
           'pts': {'metro': np.round(city['pts']['метро'], 2).tolist(), 'stop': np.round(city['pts']['остановка'], 2).tolist()}}
    if walk_max_minutes is None:
        walk_max_minutes = max(DEFAULT_PREPARED_MINUTES, walk_minutes)
    walk, reason = load_walk_access(cfg, t['block_id'], walk_minutes, walk_max_minutes, keep)
    web['access_method'] = 'walk' if walk is not None and prefer_walk else 'geometry'
    if walk is not None:
        web['walk_access'] = walk
    else:
        web['walk_access_note'] = 'Нет подходящей пешеходной матрицы. Используется геометрическое расстояние.'
        if prefer_walk:
            web['walk_access_note'] += ' ' + reason
    if pdat:
        prov = {k: pdat[k] for k in ('blocksnet', 'граф', 'max_depth') if k in pdat}
        for key in ('school', 'kindergarten'):
            sd = pdat[key]
            m = lambda a: [a[k] for k in keep]
            prov[key] = {'norm': sd['norm'], 'threshold': sd['threshold'], 'impute_capacity': sd.get('impute_capacity'),
                         'capacity': m(sd['capacity']), 'demand': m(sd['demand']), 'within': m(sd['within']),
                         'without': m(sd['without']), 'free': m(sd['free']),
                         'links': [[new[a], new[b]] + l[2:] for l in sd['links'] for a, b in [l[:2]] if a in new and b in new],
                         'pairs': [[new[a], new[b], tm] for a, b, tm in sd['pairs'] if a in new and b in new]}
        web['provision'] = prov
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(web, f, ensure_ascii=False, separators=(',', ':'))
    return path

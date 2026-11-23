"""Запуск из корня проекта: python results/final_experiments/reproduce.py."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mixuse import export, provision_bn as bn
from mixuse.blocks import build_table, load_layers, service_files
from mixuse.compare import compare
from mixuse.config import load_config
from mixuse.params import SCENARIO_ORIGINS, paired_scenarios
from mixuse.walk_access import load_walk_access

OUT = Path(__file__).resolve().parent
TERRITORIES = {'vysotsk': [305, 309, 314], 'gatchina_layers': [322, 323, 377],
               'spb_full': [8346, 9366, 2059]}
SCENARIOS = paired_scenarios()
BASE = 'Как есть'
METRICS = {'N': 'Жители территории', 'J': 'Рабочие места территории', 'K': 'Рабочих мест на жителя в окружении',
           'sP': 'Обеспеченность школами окружения, %', 'kP': 'Обеспеченность садами окружения, %'}
ACCESS = [f'{group}: {direction}' for group in ('К', 'ОД') for direction in ('получили', 'потеряли')]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def file_info(path):
    path = Path(path)
    return {'файл': path.relative_to(ROOT).as_posix(), 'sha256': bn.file_sha256(path)}


def maps(city, detail, folder, title, radius, include_baseline):
    folder.mkdir(exist_ok=True)
    indexes = [*detail['T'], *detail['E']]
    points = np.vstack([p[0] for i in indexes for p in city['polys'][i]])
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((hi - lo) * .05, 20)
    for name in ([BASE] if include_baseline else []) + list(SCENARIOS):
        fig = export.map_figure(city, detail, name, radius, title)
        ax = fig.axes[0]
        ax.set_xlim(lo[0] - padding[0], hi[0] + padding[0]); ax.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
        ax.set_aspect('equal')
        fig.savefig(folder / export.scenario_map_name(name, SCENARIOS), dpi=150, bbox_inches='tight')
        plt.close(fig)


def run(code, territory):
    print(code, 'слои', flush=True)
    cfg = load_config(ROOT / 'configs' / f'{code}.yaml')
    cfg['scenarios'] = SCENARIOS
    p = dict(cfg['params'], r=500)
    city = build_table(load_layers(cfg, cfg['_groups']), cfg, p, cfg['_groups'])
    ids = city['table']['block_id'].tolist()
    _, info, det = compare(city, territory, p, SCENARIOS)
    walk, walk_note = load_walk_access(cfg, ids, 10)
    print(code, 'обеспеченность', flush=True)
    prov = None
    try:
        version = bn.require_blocksnet()
        mx, _ = bn.load_matrix(cfg, ids)
        env = city['table']['block_id'].iloc[det['E']].astype(int).tolist()
        prov = bn.variant_provision(city, mx, territory, env, det['values'])
        school_mode = f'Точный BlocksNet {version}, весь набор, раздел 12'
        school_note = prov['_source']
        del mx
    except bn.BlocksnetError as exc:
        school_mode = 'Резервный расчёт 4.7.3, без BlocksNet'
        school_note = str(exc)
    print(code, school_mode, flush=True)

    geom_table, info, geom = compare(city, territory, p, SCENARIOS, prov=prov)
    modes = [('геометрия', geom_table, geom)]
    if walk is not None:
        table, _, network = compare(city, territory, p, SCENARIOS, prov=prov, walk_access=walk)
        modes.append(('пешком', table, network))
    folder = OUT / code
    folder.mkdir(exist_ok=True)
    rows, deltas, access_rows = [], [], []
    for name in [BASE, *SCENARIOS]:
        values = geom['values'][name]
        base = geom['values'][BASE]
        identity = {'Город': cfg['title'], 'Кварталы': ', '.join(map(str, territory)), 'КИТ': info['КИТ варианта'],
                    'Сценарий': name, 'Происхождение': SCENARIO_ORIGINS.get(name, 'исходное состояние'),
                    'Школы и сады: режим': school_mode}
        vals = {label: values[key] for key, label in METRICS.items()}
        delta = {label.replace(', %', ', п. п.'): (values[key] - base[key] if values[key] is not None and base[key] is not None else None)
                 for key, label in METRICS.items()}
        rows.append({**identity, **vals})
        deltas.append({**identity, **delta})
        for mode, _, detail in modes:
            o = detail['values'][name]
            assert all(o[key] == values[key] for key in METRICS)
            assert o['acc'].get('П') == values['acc'].get('П')
            changes = [v for g in ('К', 'ОД') for v in o['acc'].get(g, (None, None))]
            access_rows.append({**identity, 'Способ расчёта К и ОД': detail['access_method'], **dict(zip(ACCESS, changes))})
            rows[-1].update({f'{mode}: {key}': val for key, val in zip(ACCESS, changes)})
            deltas[-1].update({f'{mode}: {key}': val for key, val in zip(ACCESS, changes)})
    for filename, data in [('показатели.csv', rows), ('изменения_к_исходному.csv', deltas), ('доступность_сравнение.csv', access_rows)]:
        export.table_csv(pd.DataFrame(data), folder / filename, index=False)
    print(code, 'CSV и карты', flush=True)
    for mode, table, detail in modes:
        export.table_csv(table, folder / f'сравнение_{mode}.csv')
        export.blocks_geojson(city, folder / f'кварталы_{mode}.geojson', detail, SCENARIOS)
        maps(city, detail, folder / f'карты_{mode}', f'{cfg["title"]} · {detail["access_method"]}', p['R'],
             mode == 'геометрия')

    input_paths = [cfg[k] for k in ('blocks', 'buildings', 'zones') if cfg.get(k)]
    input_paths += [path for path, _ in service_files(cfg['services'])]
    matrix_paths = [path for path in bn.matrix_paths(cfg) if Path(path).is_file()]
    save_json(folder / 'параметры.json', {
        'город': cfg['title'], 'территория': territory, 'конфигурация': file_info(cfg['_path']), 'КИТ': info['КИТ варианта'],
        'источник КИТ': info['КИТ: откуда'], 'площадь, га': info['площадь территории, га'],
        'кварталов набора': len(ids), 'кварталов окружения': len(geom['E']), 'параметры': p, 'сценарии': SCENARIOS,
        'доступ': [d['access_method'] for _, _, d in modes], 'ограничение доступности': walk_note,
        'школы и сады': school_mode, 'подробности режима': school_note,
        'пешеходная матрица': [file_info(path) for path in matrix_paths],
        'исходные слои': [file_info(path) for path in input_paths],
        'сведения о данных': city['report'], 'предупреждения': info['предупреждения'],
    })
    print(code, 'готово', flush=True)
    return rows, deltas, access_rows


if __name__ == '__main__':
    totals = ([], [], [])
    unavailable = {}
    for code, territory in TERRITORIES.items():
        try:
            for total, entries in zip(totals, run(code, territory)):
                total.extend(entries)
        except FileNotFoundError as exc:
            unavailable[code] = str(exc)
            print(code, exc, flush=True)
    for name, data in zip(('сводная_таблица.csv', 'сводная_изменения.csv', 'сводная_доступность.csv'), totals):
        if data:
            export.table_csv(pd.DataFrame(data), OUT / name, index=False)
    if unavailable:
        save_json(OUT / 'недоступные_наборы.json', unavailable)

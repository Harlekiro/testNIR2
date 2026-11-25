"""Правила v5 для зданий (раздел 4 ТЗ): категории зданий и рабочие места в существующей застройке.

Здесь только формулы — без чтения файлов и печати. Подготовка зданий (площадь, этажность, привязка
к кварталам, встроенные помещения) — в blocks.table_from_layers; эта часть вызывается оттуда при rules: v5.

Категории (4.4):  А — функция установлена (шаги 1–2) и встроенные помещения (4.4.5);
                  Б — вероятное рабочее место (шаги 4, 5, 7);  В — без рабочих мест (шаги 3, 6, 8).
"""
from collections import defaultdict

import numpy as np

from .geometry import roundness
from .groups import main_norm
from .employment import JOB_GROUPS

STEP_NAMES = {
    1: '1 (сервисы К, ОД, П)', 2: '2 (социальные объекты)', 3: '3 (сооружения в производственных зонах)',
    4: '4 (производственная зона, от 2 этажей)', 5: '5 (производственная зона, одноэтажные от порога площади)',
    6: '6 (производственная зона, одноэтажные меньше порога)', 7: '7 (общественно-деловая зона)', 8: '8 (прочее)',
}
STEP_CATEGORY = {1: 'А', 2: 'А', 3: 'В', 4: 'Б', 5: 'Б', 6: 'В', 7: 'Б', 8: 'В'}


def building_jobs_v5(floor_area, footprint, storeys, poly, zone_group, types, social, P, norms, type_group):
    """Шаги 1–8 правила 4.4 для одного нежилого здания (S ≥ S_min). Первое выполненное условие — категория.

    floor_area A = S · этажность (4.2.1); types — типы сервисов групп К, ОД, П в контуре (точка в контуре, 4.4.2);
    social — список (тип, число мест) социальных объектов в контуре (типы из social_norms);
    zone_group — 'П', 'ОД' или другое (зона квартала, 4.4.1); type_group — тип → группа (для нормы по умолчанию).

    Возвращает (шаг, рабочие места, сведения для отчёта):
      1: A ÷ норма типа (4.4.3: тип, точек которого больше; при равенстве — большая норма; тип без нормы → a_группы);
      2: школа — число мест ÷ 13,5 (без мест — 0); иначе A ÷ норма социального типа (правило 4.4.3 по типам);
      3: производственная зона, контур круглый (4πS/P² ≥ round_threshold, вершин ≥ round_min_vertices) — 0;
      4: производственная зона, этажность ≥ prod_min_storeys — A ÷ a_prod_zone;
      5: производственная зона, площадь застройки ≥ prod_min_footprint_one_storey — A ÷ a_prod_one_storey;
      6: производственная зона, остальные — 0;   7: общественно-деловая зона — A ÷ a_business_zone;   8: прочее — 0.
    """
    info = {}
    if types:
        norm, no_norm = main_norm(types, norms, lambda ty: P['a_' + type_group[ty]])
        info['типов без нормы'] = no_norm
        return 1, floor_area / norm, info
    if social:
        SOC = P['social_norms']
        schools = [cap for ty, cap in social if ty == 'school']
        if schools:
            caps = [c for c in schools if c == c and c > 0]           # c == c — не NaN
            info['школа без числа мест'] = int(not caps)
            info['школа и другие социальные объекты'] = int(len(set(ty for ty, _ in social)) > 1)
            return 2, (sum(caps) / SOC['school'] if caps else 0.0), info
        norm, _ = main_norm([ty for ty, _ in social], SOC, lambda ty: SOC[ty])
        info['несколько социальных типов'] = int(len(set(ty for ty, _ in social)) > 1)
        return 2, floor_area / norm, info
    if zone_group == 'П':
        rd, nv = roundness(poly)
        if rd >= P['round_threshold'] and nv >= P['round_min_vertices']:
            return 3, 0.0, info
        if storeys >= P['prod_min_storeys']:
            return 4, floor_area / P['a_prod_zone'], info
        if footprint >= P['prod_min_footprint_one_storey']:
            return 5, floor_area / P['a_prod_one_storey'], info
        return 6, 0.0, info
    if zone_group == 'ОД':
        return 7, floor_area / P['a_business_zone'], info
    return 8, 0.0, info


def builtin_jobs_v5(b_area, types, group, P, norms):
    """Рабочие места встроенной площади группы g дома (4.3): B_g ÷ норма типа (4.4.3) по сервисам этой группы в доме;
    тип без нормы → a_g. Возвращает (рабочие места, число типов без нормы)."""
    if b_area <= 0 or not types:
        return 0.0, 0
    norm, no_norm = main_norm(types, norms, lambda ty: P['a_' + group])
    return b_area / norm, no_norm


def jobs_v5(bdf, bpolys, nr_idx, storeys, types_in, social_in, zone_of_block, bi_types, P, norms, type_group):
    """Рабочие места всех зданий по правилам v5. Добавляет в bdf столбцы step, jobs_A, jobs_B, category.

    Возвращает счётчики допущений и рабочие места по моделируемым функциям (А+Б и А)."""
    counters = defaultdict(int)
    steps, jA, jB, jBI = np.zeros(len(bdf), int), np.zeros(len(bdf)), np.zeros(len(bdf)), np.zeros(len(bdf))
    fa, fp, blk = bdf['floor_area'].to_numpy(), bdf['footprint'].to_numpy(), bdf['block'].to_numpy()
    use = bdf['group'].to_numpy()
    group_index = {group: i for i, group in enumerate(JOB_GROUPS)}
    by_use = np.zeros((len(bdf), len(JOB_GROUPS), 2))
    st = np.asarray(storeys, float)
    for i in nr_idx:
        k, jobs, info = building_jobs_v5(fa[i], fp[i], st[i], bpolys[i], zone_of_block[blk[i]],
                                         types_in.get(i, []), social_in.get(i, []), P, norms, type_group)
        steps[i] = k
        if STEP_CATEGORY[k] == 'А':
            jA[i] = jobs
        else:
            jB[i] = jobs
        # Отдельные социальные объекты остаются в общей занятости, но не калибруют К, ОД и П.
        if k in (1, 4, 5, 7):
            by_use[i, group_index[use[i]]] = (jobs, jA[i])
        for key, v in info.items():
            counters[key] += v
    for i, tt in bi_types.items():                       # встроенные помещения — категория А (4.4.5)
        for g in ('К', 'ОД'):
            j, no = builtin_jobs_v5(bdf.at[i, 'bi_' + g], tt[g], g, P, norms)
            jA[i] += j
            jBI[i] += j
            by_use[i, group_index[g]] += j
            counters['типов без нормы'] += no
    bdf['step'], bdf['jobs_A'], bdf['jobs_B'], bdf['jobs_builtin'] = steps, jA, jB, jBI
    bdf['category'] = [STEP_CATEGORY.get(k, '') for k in steps]
    return dict(counters), by_use


def steps_report(bdf):
    """Разбивка по шагам таблицы 4.4: зданий, поэтажная площадь, рабочие места; встроенные помещения отдельно."""
    rows = {}
    for k in range(1, 9):
        m = (bdf['step'] == k).to_numpy()
        rows[STEP_NAMES[k]] = (int(m.sum()), float(bdf.loc[m, 'floor_area'].sum()),
                               float((bdf.loc[m, 'jobs_A'] - bdf.loc[m, 'jobs_builtin'] + bdf.loc[m, 'jobs_B']).sum()))
    bi = (bdf['bi_К'] + bdf['bi_ОД']) > 0
    rows['встроенные помещения'] = (int(bi.sum()), float((bdf['bi_К'] + bdf['bi_ОД']).sum()), float(bdf['jobs_builtin'].sum()))
    return rows

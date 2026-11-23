"""Проверка входных данных и проверки чувствительности (шаги 7 и 7б)."""
import os

import pandas as pd

from .geometry import read_layer, describe_crs
from .compare import compare
from .scenarios import validate_scenarios
from .blocks import table_from_layers

_MERCATOR = (3857, 900913, 3395)


# ================================================================================================
# ПРОВЕРКА ДАННЫХ (команда check)
# ================================================================================================

def _layer_info(path):
    df, geoms, geographic = read_layer(path)
    kinds = pd.Series([g[0] for g in geoms], dtype=object).value_counts().to_dict()
    return df, kinds, geographic, df.attrs.get('epsg')


def _service_files(src):
    if os.path.isdir(src):
        return [os.path.join(src, x) for x in sorted(os.listdir(src)) if x.lower().endswith(('.geojson', '.gpkg'))]
    return [src]


def check_config(cfg):
    """Проверяет настройки и слои до расчёта. Возвращает (ошибки, предупреждения, сведения).

    Ошибки останавливают расчёт; предупреждения — нет, но их стоит прочитать.
    """
    err, warn, info = [], [], []
    err += validate_scenarios(cfg.get('scenarios'))
    fields = cfg.get('fields') or {}

    def need_file(key, label):
        p = cfg.get(key)
        if not p:
            err.append(f'Не указан слой «{label}» ({key}:).')
            return None
        if not os.path.exists(p):
            err.append(f'Слой «{label}» не найден: {p}')
            return None
        return p

    crs = {}
    # --- кварталы
    p = need_file('blocks', 'кварталы')
    if p:
        df, kinds, geo, epsg = _layer_info(p)
        crs['кварталы'] = (epsg, geo)
        info.append(f'кварталы: {len(df)} объектов ({kinds.get("poly", 0)} полигонов), {describe_crs(epsg, geo)}')
        if not kinds.get('poly'):
            err.append('В слое кварталов нет полигонов.')
        idf = cfg.get('block_id_field')
        if idf and idf not in df:
            warn.append(f'В слое кварталов нет поля номера «{idf}» — номер квартала будет равен порядку в файле (0, 1, 2…).')
        elif not idf:
            info.append('номер квартала — порядок в файле (0, 1, 2…), поле номера не задано')
        elif df[idf].duplicated().any():
            err.append(f'Номера кварталов в поле «{idf}» повторяются.')
    # --- зоны
    if cfg.get('zones'):
        p = need_file('zones', 'зоны')
        if p:
            df, kinds, geo, epsg = _layer_info(p)
            crs['зоны'] = (epsg, geo)
            info.append(f'зоны: {len(df)} объектов, {describe_crs(epsg, geo)}')
            zf = cfg.get('zone_field')
            if zf not in df:
                err.append(f'В слое зон нет поля «{zf}» (zone_field). Есть поля: {", ".join(map(str, df.columns))}')
            else:
                codes = sorted(df[zf].dropna().astype(str).unique())
                rules = cfg.get('zone_rules') or []
                if not rules and not cfg.get('_zone_dict'):
                    warn.append('Правила зон не заданы (zone_rules:): все нежилые здания без сервисов получат группу X.')
                for prefix, grp in rules:
                    if not any(c.startswith(prefix) for c in codes):
                        warn.append(f'Правило зон «{prefix}… → {grp}» не подходит ни к одному коду зоны.')
                from .groups import zone_match
                hit = {c: zone_match(c, rules, cfg.get('_zone_dict')) for c in codes}
                info.append('зона → группа нежилого здания без сервисов (X — прочее):\n      ' + '\n      '.join(
                    f'{c} → {g}' + (f' ({how})' if how else '') for c, (g, how, _) in hit.items()))
    elif cfg.get('block_zone_field'):
        info.append(f'зона квартала — поле «{cfg["block_zone_field"]}» слоя кварталов')
    else:                                                     # ТЗ 18.5: без зон расчёт не идёт
        err.append('Нет зон: укажите в анкете слой зон (землепользование — ПЗЗ или функциональные зоны генплана: файл и поле '
                   'кода) или поле зоны в слое кварталов (поле_в_кварталах). Без зон программа не отличит производственные '
                   'и общественно-деловые кварталы.')
    # --- здания и сервисы
    p = need_file('buildings', 'здания')
    if p:
        df, kinds, geo, epsg = _layer_info(p)
        crs['здания'] = (epsg, geo)
        info.append(f'здания: {len(df)} объектов ({kinds.get("poly", 0)} полигонов), {describe_crs(epsg, geo)}')
        if not fields.get('is_living') or fields['is_living'] not in df:
            err.append(f'В слое зданий нет поля «жилое» («{fields.get("is_living")}», fields.is_living). '
                       f'Есть поля: {", ".join(map(str, df.columns))}')
        for key, label, default in [('footprint', 'площадь застройки', 'будет считаться по контуру здания'),
                                    ('storeys', 'этажность', f'все здания — {cfg["params"]["storeys_default"]} этаж'),
                                    ('living_area', 'жилая площадь', 'оценка A·k_liv'),
                                    ('population', 'жители', 'оценка жилая площадь ÷ q')]:
            fld = fields.get(key)
            if fld and fld not in df:
                warn.append(f'В слое зданий нет поля «{fld}» ({label}, fields.{key}): {default}.')
            elif not fld:
                warn.append(f'Поле «{label}» не указано (fields.{key}): {default}.')
    srcs = cfg.get('services')
    srcs = srcs if isinstance(srcs, (list, tuple)) else ([srcs] if srcs else [])
    if not srcs:
        err.append('Не указаны сервисы (services:).')
    groups = cfg['_groups']
    for src in srcs:
        if not os.path.exists(src):
            err.append(f'Сервисы не найдены: {src}')
            continue
        files = _service_files(src)
        if not files:
            err.append(f'В папке сервисов {src} нет файлов .geojson / .gpkg.')
            continue
        types = []
        for fp in files:
            df, kinds, geo, epsg = _layer_info(fp)
            crs[f'сервисы {os.path.basename(fp)}'] = (epsg, geo)
            types.append(os.path.splitext(os.path.basename(fp))[0])
        if os.path.isdir(src):
            used = [t for t in types if groups.get(t, 'нет') != 'нет']
            none = [t for t in types if t not in groups]
            ignored = [t for t in types if groups.get(t) == 'нет']
            info.append(f'сервисы {src}: {len(files)} файлов; с группой: '
                        + (', '.join(f'{t} → {groups[t]}' for t in used) or 'нет'))
            if none:
                warn.append(f'Типов нет в таблице групп (не учитываются): {", ".join(none)}. '
                            f'Чтобы учесть — добавьте строки в таблицу групп (тип; название; код_ВРИ; группа).')
            if ignored:
                info.append('не учитываются по таблице групп (группа «нет»): ' + ', '.join(ignored))
            for t in [x for x in types if groups.get(x) in ('школа', 'детский_сад')]:
                fp = [x for x in files if os.path.splitext(os.path.basename(x))[0] == t]
                if fp:
                    df, *_ = _layer_info(fp[0])
                    capf = cfg.get('capacity_field', 'capacity')
                    if not capf or capf not in df:
                        warn.append(f'У объектов «{t}» нет поля числа мест «{capf}»: принято 0 мест.')
                    else:
                        n = int(pd.to_numeric(df[capf], errors='coerce').isna().sum())
                        if n:
                            warn.append(f'У {n} объектов «{t}» не указано число мест: принято 0.')
        else:
            fld = cfg.get('service_type_field')
            flist = fld if isinstance(fld, (list, tuple)) else [fld]
            if fld is None or not any(x in _layer_info(src)[0] for x in flist):
                err.append(f'В файле сервисов {src} нет поля типа {fld} (service_type_field:).')
    # --- системы координат
    systems = {}
    for name, (epsg, geo) in crs.items():
        if not geo and epsg in _MERCATOR:
            err.append(f'Слой «{name}» в EPSG:{epsg} (веб-Меркатор) — площади и расстояния искажены. '
                       f'Перепроецируйте в QGIS в EPSG:4326 или в зону UTM.')
        systems.setdefault(describe_crs(epsg, geo), []).append(name)
    if len(systems) > 1:
        bad = [k for k in systems if not (k.startswith('градусы') or k.startswith('UTM'))]
        msg = 'Слои в разных системах координат: ' + '; '.join(
            f'{k} — {", ".join(v[:3])}{" и ещё " + str(len(v) - 3) if len(v) > 3 else ""}' for k, v in systems.items())
        if bad:
            err.append(msg + '. Программа пересчитывает только градусы и зоны UTM; остальные слои перепроецируйте в QGIS.')
        else:
            err.append(msg + '. По правилу 4.1.2 все слои набора должны быть в одной системе координат: '
                       'сохраните их в QGIS в системе кварталов («Экспорт» → «Сохранить объекты как…»).')
    # --- параметры и территория
    for k, v in cfg['params'].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
            err.append(f'Параметр {k} = {v}: должен быть неотрицательным.')
    if not cfg.get('territory'):
        warn.append('Территория не задана (territory:). Команда compare попросит её указать.')
    return err, warn, info


# ================================================================================================
# ШАГ 7. Проверка устойчивости: R = 500, 800, 1000 м; КИТ = 1,4; 2,0
# ================================================================================================

def sensitivity(city, T, P, SCN, Rs=(500, 800, 1000), kits=(1.4, 2.0)):
    """Шаг 7. Сравнение повторяется при других R и КИТ: K окружения «как есть», изменение K
    в каждом варианте (K варианта − K как есть) и нехватка мест в школах.
    Таблица — как results/<код>_устойчивость.csv. Порядка вариантов нет."""
    rows = []
    for R in Rs:
        for kit in kits:
            tb, inf, _ = compare(city, T, dict(P, R=R, kit=kit), SCN)
            k = tb.loc['K окружения'].astype(float)
            sch = tb.loc['Школы: без свободного места в окружении', list(SCN)]   # 4.8.1: дети без места (4.7.0)
            rows.append({'R, м': R, 'КИТ': kit, 'кварталов в окружении': inf['кварталов в окружении'],
                         'K окружения как есть': k['Как есть'],
                         **{f'изменение K {s}': round(k[s] - k['Как есть'], 2) for s in SCN},
                         **{f'школы: без свободного места {s}': sch[s] for s in SCN}})
    return pd.DataFrame(rows)


# ================================================================================================
# ШАГ 7б. Встроенные помещения: без них и при s0 = 50, 100, 300 м²
# ================================================================================================

def builtin_sensitivity(L, cfg, P, T, SCN, s0_values=(50, 100, 300)):
    """Шаг 7б. Таблица кварталов и сравнение пересчитываются без встроенных помещений и при разных s0.

    Таблица — как results/<код>_встроенные.csv.
    Имя столбца «K окружения Как» сохранено для совместимости с эталонами ноутбука.
    """
    rows = []
    runs = [('без встроенных', dict(P, builtin=False))] + [(f's0 = {s0} м²', dict(P, builtin_s0=s0)) for s0 in s0_values]
    for name, pp in runs:
        c2 = table_from_layers(L, cfg, pp); t2 = c2['table']
        tb, inf, _ = compare(c2, T, pp, SCN)
        rows.append({'расчёт': name,
                     'встроенная площадь К, м²': int(round(t2['встр_К'].sum())), 'встроенная площадь ОД, м²': int(round(t2['встр_ОД'].sum())),
                     'рабочих мест': int(round(t2['jobs'].sum())), 'K_ref': round(float(t2['jobs'].sum() / t2['pop'].sum()), 3),
                     **_k_rows(tb, SCN, 'K окружения')})
    return pd.DataFrame(rows)


# ================================================================================================
# ПРОВЕРКИ 4.8.3–4.8.5
# ================================================================================================

def _k_rows(tb, SCN, row):
    baseline = 'Как есть' if 'Как' in SCN else 'Как'
    return {f'K окружения {baseline}': tb.loc[row, 'Как есть'],
            **{f'K окружения {s}': tb.loc[row, s] for s in SCN}}


def estimates_check(city, T, P, SCN):
    """4.8.3. Основная оценка (А + Б) против нижней границы (только А): рабочие места всего и на территории,
    K окружения «как есть» и по вариантам в обеих оценках."""
    t = city['table']
    tb, info, _ = compare(city, T, P, SCN)
    rows = []
    for name, total, row_J, row_K in [
            ('основная (А + Б)', t['jobs'].sum(), 'Рабочие места на территории J', 'K окружения'),
            ('нижняя граница (только А)', t['jobs_A'].sum(), 'Рабочие места на территории J, нижняя граница',
             'K окружения, нижняя граница')]:
        rows.append({'оценка': name, 'рабочих мест всего': int(round(total)),
                     'рабочих мест на территории как есть': tb.loc[row_J, 'Как есть'], **_k_rows(tb, SCN, row_K)})
    return pd.DataFrame(rows)


def stats_check(city, P):
    """4.8.4. Сравнение со статистикой (только если задан stat_jobs): основная оценка, нижняя граница и внешнее число."""
    if not P.get('stat_jobs'):
        return None
    t = city['table']; S = float(P['stat_jobs'])
    rows = [{'что': 'основная оценка (А + Б)', 'рабочих мест': int(round(t['jobs'].sum()))},
            {'что': 'нижняя граница (только А)', 'рабочих мест': int(round(t['jobs_A'].sum()))},
            {'что': 'статистика (stat_jobs)', 'рабочих мест': int(round(S))}]
    for r in rows:
        r['к статистике, %'] = round(100 * r['рабочих мест'] / S, 1)
    return pd.DataFrame(rows)


def threshold_check(L, cfg, P, T, SCN, values=(500, 1000, 2000)):
    """4.8.5. Порог площади одноэтажного здания производственной зоны (prod_min_footprint_one_storey):
    рабочие места шага 5, всего, K окружения «как есть» и по вариантам. Слои не перечитываются."""
    rows = []
    for v in values:
        pp = dict(P, prod_min_footprint_one_storey=v)
        c2 = table_from_layers(L, cfg, pp)
        b = c2['detail']['buildings']
        tb, _, _ = compare(c2, T, pp, SCN)
        rows.append({'порог, м²': v, 'зданий шага 5': int((b.step == 5).sum()),
                     'рабочих мест шага 5': int(round(b.loc[b.step == 5, 'jobs_B'].sum())),
                     'рабочих мест всего': int(round(c2['table']['jobs'].sum())),
                     **_k_rows(tb, SCN, 'K окружения')})
    return pd.DataFrame(rows)

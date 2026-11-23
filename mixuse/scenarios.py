"""Варианты сочетания функций: доли поэтажной площади по группам Ж, К, ОД, П; перебор сочетаний (ТЗ, раздел 7)."""
import numpy as np
import pandas as pd

GROUPS = ['Ж', 'К', 'ОД', 'П']
RESERVED_SCENARIO_NAMES = frozenset({'Как есть', '_source', '_warnings'})


def normalize(shares):
    """Доли варианта, нормированные к 1:  d_g = доля_g / Σ долей (Ж, К, ОД, П)."""
    tot = sum(shares.values())
    return {k: v / tot for k, v in shares.items()}


def validate_scenarios(scn):
    """Проверка вариантов из настроек. Возвращает список ошибок (пустой — всё верно)."""
    errors = []
    if not scn:
        return ['Не задано ни одного варианта (раздел scenarios:).']
    for name, sh in scn.items():
        if name in RESERVED_SCENARIO_NAMES:
            errors.append(f'Название варианта «{name}» зарезервировано.')
        if not isinstance(sh, dict):
            errors.append(f'Вариант «{name}»: нужны доли вида {{Ж: 60, К: 10, ОД: 30, П: 0}}.')
            continue
        extra = set(sh) - set(GROUPS)
        miss = set(GROUPS) - set(sh)
        if extra:
            errors.append(f'Вариант «{name}»: неизвестные группы {sorted(extra)}; допустимо Ж, К, ОД, П.')
        if miss:
            errors.append(f'Вариант «{name}»: не указаны доли {sorted(miss)} (поставьте 0).')
        vals = list(sh.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in vals):
            errors.append(f'Вариант «{name}»: доли должны быть неотрицательными числами.')
        elif sum(vals) <= 0:
            errors.append(f'Вариант «{name}»: сумма долей равна нулю.')
    return errors


# ================================================================================================
# ПЕРЕБОР СОЧЕТАНИЙ (ТЗ, раздел 7; команда explore)
# ================================================================================================

EXPLORE_DEFAULTS = dict(groups=['Ж', 'К', 'ОД'], step=10, max={'К': 20}, min={})


def enumerate_shares(groups=('Ж', 'К', 'ОД'), step=10, max_share=None, min_share=None):
    """Все сочетания долей допустимых групп с шагом step (%), дающие в сумме 100 %;
    доля группы — не меньше min_share[g] и не больше max_share[g]. Группы вне списка — 0.
    Порядок — по долям (Ж, затем К, ОД, П по убыванию), а не по показателям.
    Возвращает список словарей {Ж, К, ОД, П} в процентах."""
    groups = [g for g in GROUPS if g in groups]
    if not groups:
        raise ValueError('Перебор: не указано ни одной допустимой группы (Ж, К, ОД, П).')
    if step <= 0 or 100 % step:
        raise ValueError(f'Перебор: шаг {step} % должен делить 100 без остатка (например, 5, 10, 20, 25).')
    lo = {g: (min_share or {}).get(g, 0) for g in groups}
    hi = {g: (max_share or {}).get(g, 100) for g in groups}
    out = []

    def rec(i, left, cur):
        g = groups[i]
        if i == len(groups) - 1:
            if lo[g] <= left <= hi[g]:
                out.append(dict(cur, **{g: left}))
            return
        for v in range(min(hi[g], left) // step * step, -1, -step):
            if v >= lo[g]:
                rec(i + 1, left - v, dict(cur, **{g: v}))

    rec(0, 100, {})
    return [{g: c.get(g, 0) for g in GROUPS} for c in out]


def combo_name(sh):
    """Имя сочетания: «Ж70 К10 ОД20» (только ненулевые доли)."""
    return ' '.join(f'{g}{v}' for g, v in sh.items() if v) or '—'


EXPLORE_COLUMNS = [
    ('Жители на территории N', lambda o: round(o['N'])),
    ('Рабочие места на территории J', lambda o: round(o['J'])),
    ('Изменение рабочих мест к нынешним', lambda o: round(o['dJ'])),
    ('J, % рабочих мест всего набора', lambda o: round(o['Jcity'], 1)),
    ('K окружения', lambda o: round(float(o['K']), 2)),
    ('K окружения, нижняя граница', lambda o: round(float(o['KA']), 2)),
    ('K по всем данным', lambda o: round(float(o['Kall']), 3)),
    ('Доступ к К: получают', lambda o: round(o['acc']['К'][0])),
    ('Доступ к К: теряют', lambda o: round(o['acc']['К'][1])),
    ('Доступ к ОД: получают', lambda o: round(o['acc']['ОД'][0])),
    ('Доступ к ОД: теряют', lambda o: round(o['acc']['ОД'][1])),
    ('Жители в радиусе r от производства: после', lambda o: round(o['acc']['П'][1])),
    ('Школы: ученики с территории', lambda o: round(o['sT'])),
    ('Школы: без свободного места в окружении', lambda o: round(o['sNo'])),
    ('Сады: дети с территории', lambda o: round(o['kT'])),
    ('Сады: без свободного места в окружении', lambda o: round(o['kNo'])),
]
_NOT_FOR_NOW = ('Изменение', 'Доступ', 'Школы: без', 'Сады: без', 'Жители в радиусе')   # у «как есть» пусто


def explore(city, T, P, combos):
    """Показатели раздела 4.7 для каждого сочетания (правила v5) — той же функцией, что для вариантов (compare_v5);
    окружение и доступ считаются один раз для всех сочетаний.
    Возвращает (таблица «сочетание × показатели», первая строка — «Как есть»; сводка compare_v5: КИТ и окружение)."""
    from .compare import compare_v5
    scn = {combo_name(c): c for c in combos}
    _, info, det = compare_v5(city, T, P, scn)
    base = det['values']['Как есть']
    rows = [{'сочетание': 'Как есть', **{f'{g}, %': '' for g in GROUPS},
             **{col: ('' if col.startswith(_NOT_FOR_NOW) else f(base)) for col, f in EXPLORE_COLUMNS}}]
    for name, sh in scn.items():
        o = det['values'][name]
        rows.append({'сочетание': name, **{f'{g}, %': sh[g] for g in GROUPS},
                     **{col: f(o) for col, f in EXPLORE_COLUMNS}})
    return pd.DataFrame(rows), info


def apply_filters(table, conditions):
    """Фильтр по условиям пользователя: «показатель>=число» (знаки >=, <=, >, <, =). Порядок строк не меняется;
    строка «Как есть» остаётся всегда."""
    import operator
    ops = [('>=', operator.ge), ('<=', operator.le), ('>', operator.gt), ('<', operator.lt), ('=', operator.eq)]
    keep = pd.Series(True, index=table.index)
    for cond in conditions or []:
        for sign, fn in ops:
            if sign in cond:
                col, val = [x.strip() for x in cond.split(sign, 1)]
                break
        else:
            raise ValueError(f'Условие «{cond}»: нужен вид «показатель>=число» (знаки >=, <=, >, <, =).')
        if col not in table:
            raise ValueError(f'Условие «{cond}»: нет показателя «{col}». Есть: {", ".join(map(str, table.columns[1:]))}.')
        try:
            num = float(val.replace(',', '.'))
        except ValueError:
            raise ValueError(f'Условие «{cond}»: «{val}» — не число.')
        keep &= fn(pd.to_numeric(table[col], errors='coerce'), num)
    keep |= table['сочетание'] == 'Как есть'
    return table[keep].reset_index(drop=True)


def _num(v):
    """Целое — без дробной части (12 313, а не 12313.000)."""
    return int(v) if float(v).is_integer() else round(float(v), 3)


def explore_summary(table, groups):
    """Сводка перебора: диапазон каждого показателя и то, доля какой группы на него влияет.

    Влияние группы g — коэффициент корреляции r между долей g и показателем по всем сочетаниям и наклон прямой:
    на сколько в среднем меняется показатель, если доля g больше на 10 п. п. (доли в сумме 100 %, поэтому рост
    одной доли — это уменьшение других). «Сильнее всего зависит от» — группа с наибольшим |r|.
    Без оценок «лучше / хуже» и без сортировки сочетаний."""
    t = table[table['сочетание'] != 'Как есть']
    rows = []
    for col, _ in EXPLORE_COLUMNS:
        y = pd.to_numeric(t[col], errors='coerce').to_numpy(float)
        r = {'показатель': col, 'от': _num(y.min()), 'до': _num(y.max())}
        best, best_r = None, 0.0
        for g in groups:
            x = pd.to_numeric(t[f'{g}, %'], errors='coerce').to_numpy(float)
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                r[f'{g}: r'] = ''
                r[f'{g}: на +10 п. п.'] = ''
                continue
            rr = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(x, y, 1)[0]) * 10
            r[f'{g}: r'] = round(rr, 2)
            r[f'{g}: на +10 п. п.'] = round(slope, 3) if abs(slope) < 1 else round(slope)
            if abs(rr) > abs(best_r):
                best, best_r = g, rr
        r['сильнее всего зависит от доли'] = ('не зависит от долей' if np.ptp(y) == 0 or best is None else
                                             f'{best} ({"растёт" if best_r > 0 else "убывает"} с ростом доли)')
        rows.append(r)
    return pd.DataFrame(rows)

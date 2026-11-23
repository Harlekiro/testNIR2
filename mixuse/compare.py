"""Шаги 5–6: окружение территории и сравнение вариантов (раздел 4.6–4.7 ТЗ).

Чистые функции: на входе — таблица кварталов (результат blocks.build_table), номера кварталов
территории, параметры и варианты; на выходе — таблица сравнения. Файлы не читаются и не пишутся.
Сводного балла и порядка вариантов нет.
"""
from .errors import UserMessage
import numpy as np
import pandas as pd

from .geometry import dist_points_to_polys
from .groups import FUNCS
from .scenarios import RESERVED_SCENARIO_NAMES, normalize
from .walk_access import walk_neighbors


class TerritoryError(UserMessage, ValueError):
    """Территория задана неверно — сообщение для пользователя."""


def territory_indices(table, T_ids):
    """Номера кварталов территории → номера строк таблицы. Понятная ошибка, если номера нет."""
    if not T_ids:
        raise TerritoryError('Территория не задана. Укажите номера кварталов в настройках (territory: [..]) '
                             'или в команде: --territory 12,13. Номера видны в слое results/<код>_кварталы.geojson, поле block_id.')
    ids = table.block_id.to_numpy()
    missing = [i for i in T_ids if i not in set(ids.tolist())]
    if missing:
        raise TerritoryError(f'Нет кварталов с номерами {missing}. Номера видны в слое results/<код>_кварталы.geojson, поле block_id.')
    return [int(np.where(ids == i)[0][0]) for i in T_ids]


def environment(city, T, R):
    """Шаг 5. Окружение E — кварталы вне T, центр которых не дальше R от контура территории.

        distT(b) = min по кварталам территории расстояния от центра b до контура (0 внутри).
    """
    t = city['table']; n = len(t)
    inT = np.zeros(n, bool); inT[T] = True
    C = t[['cx', 'cy']].to_numpy()
    distT = np.min([dist_points_to_polys(C, city['polys'][k]) for k in T], axis=0)
    E = np.where(~inT & (distT <= R))[0]
    return E, inT, distT


def compare(city, T_ids, P, SCN, prov=None, *, walk_access=None):
    """Шаги 5–6: окружение и сравнение вариантов (compare_v5). Возвращает (подробная таблица, сводка, данные)."""
    return compare_v5(city, T_ids, P, SCN, prov, walk_access=walk_access)


# ================================================================================================
# Выбор КИТ, подробная и краткая таблица, вывод (ТЗ 4.6, 4.7)
# ================================================================================================

GROUP_WORDS = {'К': 'торговле и услугам', 'ОД': 'общественно-деловой функции'}


class KitError(UserMessage, ValueError):
    """КИТ варианта не задан — сообщение для пользователя."""


def fmt(v, d=0):
    """Число с пробелом между разрядами, как на странице: 12 345."""
    return f'{v:,.{d}f}'.replace(',', ' ')


def choose_kit(t, E, P):
    """КИТ варианта — один для всех вариантов территории (правило 4.6.2, порядок 5.1):
        1) КИТ из ПЗЗ или проекта планировки — значение пользователя (kit, источник kit_source);
        2) если его нет — коэффициент плотности застройки по СП 42.13330.2016, табл. Б.1, строка kit_sp_row
           (предельное значение для квартала брутто: вариант показывает наибольший допустимый объём);
        3) предел по ПЗЗ — только проверка: КИТ_пред = pzz_build_share × pzz_max_storeys; КИТ варианта больше — предупреждение;
        4) КИТ окружения по данным — только для сравнения: КИТ_E = Σ поэтажной площади всех зданий E ÷ Σ площади кварталов E;
           если у большей части зданий окружения этажность по умолчанию — предупреждение, что КИТ_E занижен.
    Возвращает (КИТ, источник, КИТ_E, предупреждения)."""
    from .params import SP42_B1, SP42_NAME
    area = t['area'].iloc[E].sum()
    kitE = float(t['fa'].iloc[E].sum() / area) if len(E) and area > 0 else None
    warns = []
    n = t['n_bld'].iloc[E].sum() if len(E) else 0
    if n and t['n_bld_default'].iloc[E].sum() / n > 0.5:
        warns.append(f'У {100 * t["n_bld_default"].iloc[E].sum() / n:.0f} % зданий окружения этажность не указана и принята '
                     'по умолчанию (4.2.2): КИТ окружения по данным занижен.')
    kit, source = kit_value(P)
    share, storeys = P.get('pzz_build_share'), P.get('pzz_max_storeys')
    if share and storeys:
        share = share / 100 if share > 1 else share          # 40 % или 0,4
        lim = share * storeys
        if kit > lim + 1e-9:
            warns.append(f'КИТ варианта {kit:.2f} больше предела по ПЗЗ {lim:.2f} (процент застройки {100 * share:.0f} % × '
                         f'{storeys:g} этажей, 4.6.2, п. 3).')
    return kit, source, kitE, warns


def kit_value(P):
    """КИТ варианта и его источник по правилу 4.6.2 (п. 1–2): значение пользователя или строка табл. Б.1."""
    from .params import SP42_B1, SP42_NAME
    if P.get('kit') is not None:
        return float(P['kit']), 'пользователь: ' + (P.get('kit_source') or 'ПЗЗ или проект планировки (источник не указан)')
    if P.get('kit_sp_row'):
        name, kit = SP42_B1[P['kit_sp_row']]
        return kit, f'свод правил: {SP42_NAME}, строка «{name}» ({P["kit_sp_row"]})'
    rows = '; '.join(f'{k} — {n} ({v:g})'.replace('.', ',') for k, (n, v) in SP42_B1.items())
    raise KitError('КИТ варианта не задан (правило 4.6.2). Укажите КИТ из ПЗЗ или проекта планировки и его источник '
                   '(анкета: кит=dict(значение=…, источник=…); файл настроек: params kit, kit_source) или, если в ПЗЗ '
                   'его нет, строку таблицы Б.1 СП 42.13330.2016 по зоне, в которую переводится территория '
                   f'(анкета: кит=dict(строка_СП=…); файл настроек: params kit_sp_row). Строки: {rows}. '
                   'КИТ окружения по данным основным значением быть не может.')


def kit_line(kit, kitE):
    """Строка «КИТ варианта / окружения (в N раз плотнее)» (4.6.2, п. 4)."""
    if not kitE:
        return f'{kit:.2f} / — (окружения нет)'
    r = kit / kitE
    how = f'в {r:.1f} раз плотнее' if r >= 1 else f'в {1 / r:.1f} раз реже'
    return f'{kit:.2f} / {kitE:.2f} ({how})'.replace('.', ',')


def compare_v5(city, T_ids, P, SCN, prov=None, *, walk_access=None):
    """Шаги 5–6 по правилам v5. Существующая застройка территории в варианте заменяется полностью (4.6.3).

        F = S_T · КИТ;  N = F · δ_Ж · k_liv ÷ q;  J = F · (δ_К ÷ a_К + δ_ОД ÷ a_ОД + δ_П ÷ a_П)        (4.6.2, 4.6.4)
        ΔJ = J − J_T0;  K окружения = (J_E + J) ÷ (P_E + N);  нижняя граница — (J_E,A + J_A) ÷ (P_E + N);
        K по всем данным = (ΣJobs − J_T0 + J) ÷ (ΣPop − P_T0 + N)                                        (4.7.1)
        доступ к К и ОД — получают / теряют; П — жители в радиусе r от производства до / после           (4.7.2)
        школы и сады (запасной расчёт 4.7.3): свободные места = max(0; места E − потребность жителей E);
        дети без места = max(0; дети с территории − свободные места); «как есть» — обеспеченность окружения
        (места E и T ÷ потребность (P_E + P_T0)).
    prov — обеспеченность через blocksnet по вариантам (provision_bn.variant_provision, раздел 12): если задана,
    строки школ и садов — 12.5 (обеспеченность окружения в пределах доступности, дети с территории, из них обеспечены,
    изменение обеспеченности окружения, получили место вообще); без неё — запасной расчёт 4.7.3.
    Возвращает (подробная таблица, сводка, данные: краткая таблица, выводы, карта)."""
    reserved = RESERVED_SCENARIO_NAMES.intersection(SCN)
    if reserved:
        raise ValueError(f'Зарезервированные названия вариантов: {", ".join(sorted(reserved))}.')
    t = city['table']; G = city['geom']
    T = territory_indices(t, T_ids)
    E, inT, distT = environment(city, T, P['R'])
    kit, kit_from, kitE, kit_warns = choose_kit(t, E, P)
    C = t[['cx', 'cy']].to_numpy()
    S = t.area.iloc[T].sum(); F = S * kit
    popE, jobsE, jobsE_A = t['pop'].iloc[E].sum(), t['jobs'].iloc[E].sum(), t['jobs_A'].iloc[E].sum()
    popA, jobsA = t['pop'].sum(), t['jobs'].sum()
    popT0, jobsT0, jobsT0_A = t['pop'].iloc[T].sum(), t['jobs'].iloc[T].sum(), t['jobs_A'].iloc[T].sum()
    pres = {g: t['есть_' + g].to_numpy() for g in FUNCS}
    near = {b: G.dist_from_point(*C[b]) <= P['r'] for b in E if t['pop'].iloc[b] > 0}
    walk = walk_neighbors(walk_access, t['block_id'], near) if walk_access is not None else None
    Tpolys = [p for k in T for p in city['polys'][k]]
    out, effect, effect_by_function = {}, {}, {}
    for name, sh in [('Как есть', None)] + list(SCN.items()):
        if sh is None:
            N, J, J_A, d = popT0, jobsT0, jobsT0_A, None
            capS = t.cap_school.iloc[list(E) + T].sum(); capK = t.cap_kinder.iloc[list(E) + T].sum()
        else:
            d = normalize(sh)
            N = F * d['Ж'] * P['k_liv'] / P['q']
            J = F * (d['К'] / P['a_К'] + d['ОД'] / P['a_ОД'] + d['П'] / P['a_П'])
            J_A = J
            capS, capK = t.cap_school.iloc[E].sum(), t.cap_kinder.iloc[E].sum()
        acc, eff = {}, {}
        changes = {g: {} for g in ('К', 'ОД')}
        if d is not None:
            for g in FUNCS:
                a = c = 0.0
                for b, nb in near.items():
                    if walk is not None and g != 'П':
                        destinations = walk[b]
                        before = bool(pres[g][destinations].any())
                        after = bool((pres[g][destinations] & ~inT[destinations]).any()
                                     or (d[g] > 0 and inT[destinations].any()))
                    else:
                        before = bool((nb & pres[g]).any())
                        after = bool((nb & pres[g] & ~inT).any()) or (d[g] > 0 and distT[b] <= P['r'])
                    p = t['pop'].iloc[b]
                    if g == 'П':
                        a += p if before else 0.0
                        c += p if after else 0.0
                        continue
                    if not before and after:
                        a += p; eff[b] = eff.get(b, 0) + 1; changes[g][b] = 1
                    if before and not after:
                        c += p; eff[b] = eff.get(b, 0) - 1; changes[g][b] = -1
                acc[g] = (a, c)
        effect[name] = eff
        effect_by_function[name] = changes
        nS, nK = (popE + N) * P['n_school'] / 1000, (popE + N) * P['n_kinder'] / 1000
        freeS = max(0.0, capS - popE * P['n_school'] / 1000)
        freeK = max(0.0, capK - popE * P['n_kinder'] / 1000)
        out[name] = dict(N=N, J=J, J_A=J_A, d=d, dJ=None if d is None else J - jobsT0, acc=acc,
                         K=(jobsE + J) / (popE + N), KA=(jobsE_A + J_A) / (popE + N),
                         Kall=(jobsA - jobsT0 + J) / (popA - popT0 + N), Jcity=100 * J / jobsA if jobsA else 0.0,
                         Ncity=100 * N / popA if popA else 0.0,
                         sT=N * P['n_school'] / 1000, kT=N * P['n_kinder'] / 1000,
                         sP=100 * capS / nS if nS > 0 else None, kP=100 * capK / nK if nK > 0 else None,
                         sNo=max(0.0, N * P['n_school'] / 1000 - freeS), kNo=max(0.0, N * P['n_kinder'] / 1000 - freeK))
    names = list(out)
    if prov:                                   # 12.5: школы и сады — blocksnet; дети без места = дети − из них обеспечены
        for n in names:
            s_, k_ = prov[n]['school'], prov[n]['kindergarten']
            out[n].update(prov=prov[n], sP=s_['env_in'], kP=k_['env_in'], sT=s_['children'], kT=k_['children'],
                          sNo=s_['children'] - s_['provided'], kNo=k_['children'] - k_['provided'])
    to_city = city_scale(out, popA)            # ТЗ 18.2а: сравнение с городом целиком — автоматически
    full = pd.DataFrame({n: detailed_column(out[n]) for n in names})
    short = pd.DataFrame({n: short_column(out[n], out, to_city) for n in names})
    concl = {n: conclusion(out[n], out['Как есть'], P, to_city) for n in names[1:]}
    short.loc['Вывод'] = ['—'] + [concl[n] for n in names[1:]]
    method = (f"OpenStreetMap, пешком ≤ {walk_access['threshold_minutes']:g} мин"
              if walk_access is not None else f"геометрическое расстояние, r = {P['r']:g} м")
    if walk_access is not None:
        full.loc['Способ расчёта доступа к К и ОД'] = method
        short.loc['Способ расчёта доступа к К и ОД'] = method
    info = {'кварталов на территории': len(T), 'площадь территории, га': round(float(S) / 1e4, 1),
            'КИТ варианта': kit, 'КИТ: откуда': kit_from,
            'КИТ окружения по данным (для сравнения)': None if kitE is None else round(kitE, 2),
            'КИТ варианта / окружения': kit_line(kit, kitE),
            'поэтажная площадь F, м²': int(round(F)),
            'кварталов в окружении': len(E), 'жителей в окружении': int(round(popE)), 'рабочих мест в окружении': int(round(jobsE)),
            'рабочих мест в окружении, нижняя граница': int(round(jobsE_A)),
            'K_ref — рабочих мест на жителя по всем данным (справочно)': round(float(jobsA / popA), 3) if popA else None,
            'доступ к К и ОД': method,
            'школы и сады': (prov.get('_source', 'blocksnet (раздел 12)') if prov
                             else 'запасной расчёт 4.7.3: места и потребность внутри окружения')}
    if len(city['pts']['метро']):
        info['до метро, м'] = int(round(dist_points_to_polys(city['pts']['метро'], Tpolys).min()))
    if len(city['pts']['остановка']):
        info['до остановки, м'] = int(round(dist_points_to_polys(city['pts']['остановка'], Tpolys).min()))
    warns = list(kit_warns) + list((prov or {}).get('_warnings', []))
    maxJ = max(out[n]['Jcity'] for n in names[1:]) if len(names) > 1 else 0
    maxN = max(out[n]['N'] for n in names[1:]) if len(names) > 1 else 0
    if maxJ > CITY_SHARE or maxN > CITY_SHARE / 100 * popA:   # 10 % рабочих мест или жителей набора (18.2а)
        warns.append(f'Масштаб: вариант добавляет до {fmt(maxJ)} % рабочих мест всего набора и до {fmt(maxN)} жителей '
                     f'(во всём наборе {fmt(popA)}). Варианты различаются объёмом застройки сильнее, чем сочетанием функций. '
                     'Проверьте КИТ или уменьшите территорию.')
    info['предупреждения'] = warns
    return full, info, {'T': T, 'E': E, 'effect': effect, 'effect_by_function': effect_by_function,
                        'access_method': method, 'short': short, 'conclusions': concl, 'values': out, 'city': to_city}


def _acc(o, g):
    return '—' if o['d'] is None else f"+{fmt(o['acc'][g][0])} / −{fmt(o['acc'][g][1])}"


CITY_SHARE = 10.0      # %: вариант добавляет больше этой доли жителей или рабочих мест набора — сравнение с городом (18.2а)


def city_scale(out, popA):
    """Сравнение с городом целиком нужно (ТЗ 18.2а), если хотя бы один вариант добавляет больше CITY_SHARE %
    рабочих мест или жителей всего набора: малый город или большая территория."""
    names = [n for n in out if out[n]['d'] is not None]
    return any(out[n]['Jcity'] > CITY_SHARE or (popA and 100 * out[n]['N'] / popA > CITY_SHARE) for n in names)


def detailed_column(o):
    """Подробная таблица (results/<код>_сравнение_подробно.csv): строки 4.7.1–4.7.3, как на демо-странице."""
    now = o['d'] is None
    return {
        'Жители на территории N': int(round(o['N'])), 'Рабочие места на территории J': int(round(o['J'])),
        'Изменение рабочих мест к нынешним': '—' if now else int(round(o['dJ'])),
        'Доля нежилых площадей, %': '' if now else round(100 * (1 - o['d']['Ж'])),
        'J, % рабочих мест всего набора': round(o['Jcity'], 1),
        'N, % жителей всего набора': round(o['Ncity'], 1),
        'K окружения': round(float(o['K']), 2), 'K окружения, нижняя граница': round(float(o['KA']), 2),
        'Рабочие места на территории J, нижняя граница': int(round(o['J_A'])),
        'K по всем данным': round(float(o['Kall']), 3),
        'Доступ к К: получают / теряют': _acc(o, 'К'), 'Доступ к ОД: получают / теряют': _acc(o, 'ОД'),
        'Жители в радиусе r от производства: до / после': '—' if now else f"{fmt(o['acc']['П'][0])} / {fmt(o['acc']['П'][1])}",
        **(provision_rows(o) if o.get('prov') else {
            'Школы: ученики с территории': int(round(o['sT'])),
            'Школы: обеспеченность окружения, %': '' if o['sP'] is None else int(round(o['sP'])),
            'Школы: без свободного места в окружении': '—' if now else int(round(o['sNo'])),
            'Сады: дети с территории': int(round(o['kT'])),
            'Сады: обеспеченность окружения, %': '' if o['kP'] is None else int(round(o['kP'])),
            'Сады: без свободного места в окружении': '—' if now else int(round(o['kNo'])),
        }),
    }


def provision_rows(o):
    """Строки 12.5 (заменяют строки 4.7.3), если обеспеченность посчитана через blocksnet."""
    now = o['d'] is None
    rows = {}
    for key, label in [('school', 'Школы'), ('kindergarten', 'Сады')]:
        v = o['prov'][key]
        r1 = lambda x: '' if x is None else round(float(x), 1)
        rows.update({
            f'{label}: обеспеченность окружения в пределах доступности, %': r1(v['env_in']),
            f'{label}: дети с территории': '—' if now else v['children'],
            f'{label}: из них обеспечены в пределах доступности': '—' if now else v['provided'],
            f'{label}: изменение обеспеченности окружения, п. п.': '—' if now else r1(v['delta']),
            f'{label}: получили место вообще (в том числе дальше порога), %': r1(v['env_any']),
        })
    return rows


def short_column(o, out, city=False):
    """Краткая таблица 4.7.0 (основной вывод программы и страницы). city — строки сравнения с городом целиком (18.2а)."""
    now = o['d'] is None
    first = out[list(out)[1]] if len(out) > 1 else None
    if now:
        prod = fmt(first['acc']['П'][0]) if first else '—'
        school = (f"обеспеченность {'—' if o['sP'] is None else fmt(o['sP'])} % / "
                  f"{'—' if o['kP'] is None else fmt(o['kP'])} %")
    else:
        prod = fmt(o['acc']['П'][1])
        school = f"{fmt(o['sT'])} / {fmt(o['kT'])} ({fmt(o['sNo'])} / {fmt(o['kNo'])})"
    return {
        'Жители на территории': fmt(o['N']),
        'Рабочие места на территории (изменение к нынешним)':
            fmt(o['J']) if now else f"{fmt(o['J'])} ({'+' if o['dJ'] > 0 else ''}{fmt(o['dJ'])})",
        'Рабочих мест на жителя в окружении': f"{o['K']:.2f}",
        **({'K по всем данным': f"{o['Kall']:.2f}", 'J, % рабочих мест набора': fmt(o['Jcity']),
            'N, % жителей набора': fmt(o['Ncity'])} if city else {}),
        'Доступ к торговле и услугам: получают / теряют, жителей': _acc(o, 'К'),
        'Доступ к общественно-деловой функции: получают / теряют': _acc(o, 'ОД'),
        'Жители рядом с производством': prod,
        'Школьники / дошкольники с территории (без свободного места в окружении)': school,
    }


def _word(d):
    return 'почти без изменений' if abs(d) < 0.03 else ('больше' if d > 0 else 'меньше')


def conclusion(o, base, P=None, city=False):
    """Вывод (4.7.0): фразы по порядку, без оценок «лучше / хуже» и без сравнения вариантов между собой.
    city — после окружения сравнение с городом целиком (city_scale, ТЗ 18.2а);
    P['housing_note'] — примечание в конце вывода для вариантов с жильём (ТЗ 17)."""
    P = P or {}
    t = [f"Новых жителей — {fmt(o['N'])}, рабочих мест — {fmt(o['J'])}" if round(o['N']) > 0
         else f"Жилья нет, рабочих мест — {fmt(o['J'])}"]
    t.append(f"к нынешним рабочим местам {'+' if o['dJ'] >= 0 else '−'}{fmt(abs(o['dJ']))}")
    t.append(f"рабочих мест на жителя в окружении {_word(o['K'] - base['K'])}: {base['K']:.2f} → {o['K']:.2f}")
    if city:
        t.append(f"рабочих мест на жителя по городу целиком {_word(o['Kall'] - base['Kall'])}: {base['Kall']:.2f} → {o['Kall']:.2f}; "
                 f"рабочие места территории — {fmt(o['Jcity'])} % рабочих мест города"
                 + (f", жители — {fmt(o['Ncity'])} % жителей города" if round(o['N']) > 0 else ''))
    ch = []
    for g in ('К', 'ОД'):
        a, c = o['acc'][g]
        if round(a) > 0:
            ch.append(f'доступ к {GROUP_WORDS[g]} получают жители ({fmt(a)} чел.)')
        if round(c) > 0:
            ch.append(f'доступ к {GROUP_WORDS[g]} теряют жители ({fmt(c)} чел.)')
    t.append('; '.join(ch) if ch else 'доступ к торговле и общественно-деловой функции без изменений')
    pb, pa = o['acc']['П']
    if round(pa) < round(pb):
        t.append(f'рядом с производством жителей меньше на {fmt(pb - pa)}')
    elif round(pa) > round(pb):
        t.append(f'рядом с производством жителей больше на {fmt(pa - pb)}')
    if round(o['N']) > 0:
        sN, kN = round(o['sNo']), round(o['kNo'])
        t.append(f'без свободного места в окружении: школьники — {fmt(sN)}, дошкольники — {fmt(kN)}' if sN + kN > 0
                 else 'свободных мест в школах и садах окружения хватает')
    if P.get('housing_note') and round(o['N']) > 0:
        t.append(P['housing_note'])
    return '; '.join(t) + '.'

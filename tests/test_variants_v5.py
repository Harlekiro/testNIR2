"""Этап 5: варианты v5 (4.6), выбор КИТ (4.6.2, 5.1), краткая таблица и вывод (4.7.0).

Маленький пример — test_rules_v5.make_v5; территория — квартал 21. Окружение (центр не дальше 800 м от контура
квартала 21, x 300–500): кварталы 20 (центр x = 100, 200 м), 22 (700, 200 м), 23 (1 000, 500 м), 24 (1 300, ровно 800 м).
КИТ варианта — по СП 42.13330.2016, табл. Б.1 (zh_low_mid, 0,8); КИТ окружения — только для сравнения (4.6.2):
  кв. 20: 800 + 1 200 + 400 + 800 + 312,14 + 312,14 = 3 824,29;  кв. 22: 2 000 + 400 + 800 + 400 = 3 600;
  кв. 23: 800;  кв. 24: 400  →  8 624,29 ÷ (4 · 40 000) = 0,0539 → 0,05.
"""
import pytest

from test_rules_v5 import make_v5, build, CIRCLE
from mixuse.compare import compare, conclusion, choose_kit, environment, territory_indices, KitError
from mixuse.params import SCENARIOS
from mixuse.config import load_config, param_origins, ConfigError


def test_default_scenarios_v5(tmp_path):
    p = make_v5(str(tmp_path))
    text = open(p, encoding='utf-8').read()
    start = text.index('scenarios:')
    end = text.index('\n', text.index('"ОД+П"'))
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text[:start] + text[end + 1:])                # без своих вариантов → пять вариантов 4.6.1
    cfg = load_config(p)
    assert cfg['scenarios'] == SCENARIOS
    assert list(cfg['scenarios']) == ['Ж+К', 'Ж+К+ОД', 'К+ОД', 'К+ОД+П', 'Ж+К+П']
    assert cfg['scenarios']['Ж+К+ОД'] == {'Ж': 50, 'К': 15, 'ОД': 35, 'П': 0}
    assert cfg['scenarios']['К+ОД'] == {'Ж': 0, 'К': 35, 'ОД': 65, 'П': 0}

NL = chr(10)
SP_ROW = 'params: {kit_sp_row: zh_low_mid}' + NL     # КИТ по СП 42.13330.2016, табл. Б.1: 0,8


def test_kit_required(tmp_path):
    """4.6.2: без КИТ из ПЗЗ и без строки СП расчёт не идёт — КИТ окружения основным значением не служит."""
    cfg, L, city = build(make_v5(str(tmp_path)))
    with pytest.raises(KitError, match='kit_sp_row'):
        compare(city, [21], cfg['params'], cfg['scenarios'])


def test_kit_by_sp42_and_env_for_comparison(tmp_path):
    cfg, L, city = build(make_v5(str(tmp_path), extra=SP_ROW))
    t = city['table']
    T = territory_indices(t, [21])
    E, _, _ = environment(city, T, 800)
    assert sorted(t.block_id.iloc[E]) == [20, 22, 23, 24]
    fa = 800 + 1200 + 400 + 800 + 2 * CIRCLE + 2000 + 400 + 800 + 400 + 800 + 400
    kit, how, kitE, warns = choose_kit(t, E, cfg['params'])
    assert kit == 0.8 and how.startswith('свод правил: СП 42.13330.2016, табл. Б.1')
    assert kitE == pytest.approx(fa / 160000, rel=1e-4) and warns == []
    full, info, det = compare(city, [21], cfg['params'], cfg['scenarios'])
    F = 40000 * 0.8
    assert info['поэтажная площадь F, м²'] == round(F)
    assert info['КИТ окружения по данным (для сравнения)'] == 0.05
    assert info['КИТ варианта / окружения'] == f'0,80 / 0,05 (в {0.8 / (fa / 160000):.1f} раз плотнее)'.replace('.', ',')
    assert full.loc['Жители на территории N', 'Ж+К+ОД'] == round(F * 0.75 * 0.7 / 30)
    assert full.loc['Рабочие места на территории J', 'Ж+К+ОД'] == round(F * (0.05 / 28 + 0.20 / 15))
    origins = {k: how for k, _, how in param_origins(cfg, info)}
    assert origins['kit'].startswith('свод правил') and origins['R'] == 'по умолчанию — без источника'


def test_kit_user_has_priority_and_pzz_check(tmp_path):
    extra = 'params: {kit: 1.4, kit_source: ПЗЗ, kit_sp_row: zh_low_mid, pzz_build_share: 40, pzz_max_storeys: 3}' + NL
    cfg, L, city = build(make_v5(str(tmp_path), extra=extra))
    full, info, det = compare(city, [21], cfg['params'], cfg['scenarios'])
    assert (info['КИТ варианта'], info['КИТ: откуда']) == (1.4, 'пользователь: ПЗЗ')
    assert dict((k, h) for k, _, h in param_origins(cfg, info))['kit'] == 'пользователь: ПЗЗ'
    assert any('больше предела по ПЗЗ 1.20' in w for w in info['предупреждения'])      # 0,4 × 3 = 1,2 < 1,4
    t = city['table'].copy()                       # предупреждение «КИТ окружения занижен»
    t['n_bld_default'] = t['n_bld']
    T = territory_indices(t, [21])
    E, _, _ = environment(city, T, 800)
    assert any('занижен' in w for w in choose_kit(t, E, cfg['params'])[3])


def test_kit_sp_row_checked(tmp_path):
    with pytest.raises(ConfigError, match='kit_sp_row'):
        load_config(make_v5(str(tmp_path), extra='params: {kit_sp_row: zh_mega}' + NL))


def test_short_table_rows(tmp_path):
    cfg, L, city = build(make_v5(str(tmp_path), extra=SP_ROW))
    full, info, det = compare(city, [21], cfg['params'], cfg['scenarios'])
    city = ['K по всем данным', 'J, % рабочих мест набора', 'N, % жителей набора']   # 18.2а: малый пример — порог 10 % сработал
    assert det['city']
    assert list(det['short'].index) == [
        'Жители на территории', 'Рабочие места на территории (изменение к нынешним)', 'Рабочих мест на жителя в окружении',
        *city, 'Доступ к торговле и услугам: получают / теряют, жителей', 'Доступ к общественно-деловой функции: получают / теряют',
        'Жители рядом с производством', 'Школьники / дошкольники с территории (без свободного места в окружении)', 'Вывод']
    assert det['short'].loc['Вывод', 'Как есть'] == '—'
    assert 'Школы: не хватает мест' not in full.index            # 4.7.3: формула «нехватка после − как есть» не используется


def V(N, J, dJ, K, acc, sNo=0.0, kNo=0.0):
    return dict(N=N, J=J, dJ=dJ, K=K, acc=acc, sNo=sNo, kNo=kNo)


BASE = V(0, 100, None, 0.50, {})
NO = {'К': (0, 0), 'ОД': (0, 0), 'П': (10, 10)}


def test_conclusion_phrases():
    """Фразы 4.7.0 по порядку; числа — через тире или в скобках; без оценок и сравнения вариантов."""
    assert conclusion(V(1234.4, 50, -50, 0.52, NO), BASE) == (
        'Новых жителей — 1 234, рабочих мест — 50; к нынешним рабочим местам −50; '
        'рабочих мест на жителя в окружении почти без изменений: 0.50 → 0.52; '
        'доступ к торговле и общественно-деловой функции без изменений; '
        'свободных мест в школах и садах окружения хватает.')
    acc = {'К': (120, 0), 'ОД': (0, 35), 'П': (300, 100)}
    assert conclusion(V(0, 2500, 2400, 0.80, acc), BASE) == (
        'Жилья нет, рабочих мест — 2 500; к нынешним рабочим местам +2 400; '
        'рабочих мест на жителя в окружении больше: 0.50 → 0.80; '
        'доступ к торговле и услугам получают жители (120 чел.); '
        'доступ к общественно-деловой функции теряют жители (35 чел.); '
        'рядом с производством жителей меньше на 200.')
    text = conclusion(V(500, 10, -90, 0.40, {'К': (0, 0), 'ОД': (0, 0), 'П': (0, 50)}, sNo=12.4, kNo=0.2), BASE)
    assert 'меньше: 0.50 → 0.40' in text and 'рядом с производством жителей больше на 50' in text
    assert text.endswith('без свободного места в окружении: школьники — 12, дошкольники — 0.')


def test_city_scale_threshold():
    """18.2а: сравнение с городом — если вариант добавляет больше 10 % рабочих мест или жителей всего набора."""
    from mixuse.compare import city_scale
    out = lambda J, N: {'Как есть': {'d': None, 'Jcity': 50.0, 'N': 900}, 'В': {'d': {}, 'Jcity': J, 'N': N}}
    assert not city_scale(out(10.0, 100), 1000)       # ровно 10 % — нет («больше 10 %»)
    assert city_scale(out(10.1, 0), 1000)
    assert city_scale(out(0.0, 101), 1000)
    assert not city_scale(out(9.9, 100), 1000)       # «как есть» не считается

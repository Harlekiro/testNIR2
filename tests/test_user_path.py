"""Путь пользователя (ТЗ 18): анкета шага 0 вместо YAML и приёмочный тест «как пользователь» (18.4).

Приёмочный тест повторяет то, что делает пользователь по инструкции docs/инструкция_пользователя.md:
  1. скачивает программу архивом ZIP — здесь `git archive --format=zip HEAD` (так GitHub собирает «Download ZIP»;
     проверяется последний коммит), архив распаковывается в пустую папку;
  2. кладёт слои в data/<код>_user/ (tests/user_forms.py: откуда копировать);
  3. заполняет анкету шага 0 (ячейку P не трогает — ТЗ 18.2а);
  4. выполняет все ячейки ноутбука (Run All) обычным Python, в котором пакет mixuse не установлен:
     ноутбук должен сам найти папку mixuse рядом с собой;
  5. результаты results/<код>_user_*.csv совпадают с эталонами tests/etalon/v5/<код>/ (они посчитаны по YAML командой
     compare — так проверяется и 18.2: анкета и YAML дают один результат).
Петербург (около 15 минут) — только при MIXUSE_ACCEPTANCE_SPB=1.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from conftest import ROOT, ETALON_V5, same_strings
from test_etalon_v5 import notebook_reference_path, same_table, STR
from user_forms import FORMS

RUNNER = r'''
import json, sys
nb = json.load(open(sys.argv[1], encoding='utf-8'))
ns = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        exec(compile(''.join(c['source']), f'ячейка {i}', 'exec'), ns)
import mixuse
print('MIXUSE_FROM', mixuse.__file__)
'''


def zip_copy(dst):
    """Чистая копия программы, как после «Code → Download ZIP» на GitHub: архив последнего коммита."""
    z = os.path.join(dst, 'program.zip')
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=nir_code-main/', '-o', z, 'HEAD'], cwd=ROOT, check=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(dst)
    os.remove(z)
    return os.path.join(dst, 'nir_code-main')


def put_layers(code, prog):
    """Слои набора → data/<код>_user/ (как пользователь кладёт свои слои)."""
    dst = os.path.join(prog, 'data', f'{code}_user')
    for src, name in FORMS[code][2]:
        s, d = os.path.join(ROOT, src), os.path.join(dst, name)
        if not os.path.exists(s):
            pytest.skip(f'нет данных набора {code}: {src}')
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def fill_notebook(path, code):
    """Анкета шага 0 и значения в ячейке P — правка только этих двух ячеек, как делает пользователь."""
    form, params, _ = FORMS[code]
    nb = json.load(open(path, encoding='utf-8'))
    done = set()
    for c in nb['cells']:
        s = ''.join(c['source'])
        if c['cell_type'] != 'code':
            continue
        if s.startswith('# ===== ШАГ 0. ВАШИ ДАННЫЕ'):
            a, b = s.index('ДАННЫЕ = dict('), s.index('ТАБЛИЦА_ГРУПП =')
            s = s[:a] + form + s[b:]
            done.add('анкета')
        elif s.startswith('P = dict('):
            for k, v in params.items():
                s, n = re.subn(rf'^(\s*{k}=)[^,\n]+,', lambda m: m.group(1) + repr(v) + ',', s, count=1, flags=re.M)
                assert n == 1, k
            done.add('P')
        else:
            continue
        t = s.split('\n')
        c['source'] = [x + '\n' for x in t[:-1]] + ([t[-1]] if t[-1] else [])
    assert 'анкета' in done and ('P' in done or not params)
    json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def run_all(prog, nb):
    runner = os.path.join(prog, '..', 'run_all.py')
    open(runner, 'w', encoding='utf-8').write(RUNNER)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'MIXUSE_PROVISION')}
    env.update(MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, runner, nb], cwd=prog, env=env, capture_output=True, timeout=3600)
    out = r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')
    assert r.returncode == 0, out[-3000:]
    return out


@pytest.mark.parametrize('code', ['vysotsk', 'gatchina_layers', 'spb_full'])
def test_acceptance_as_user(code, tmp_path):
    if code == 'spb_full' and os.environ.get('MIXUSE_ACCEPTANCE_SPB') != '1':
        pytest.skip('Петербург — около 15 минут: MIXUSE_ACCEPTANCE_SPB=1')
    if subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True).returncode:
        pytest.skip('нет git')
    prog = zip_copy(str(tmp_path))
    assert not os.path.exists(os.path.join(prog, 'results', f'{code}_user_таблица_кварталов.csv'))
    put_layers(code, prog)
    nb = os.path.join(prog, 'метод_сравнения_вариантов.ipynb')
    fill_notebook(nb, code)
    out = run_all(prog, nb)
    assert 'MIXUSE_FROM ' + os.path.join(prog, 'mixuse') in out.replace('/', os.sep), 'mixuse взят не из скачанной папки'
    ref_dir = os.path.join(ETALON_V5, code)
    for name in sorted(os.listdir(ref_dir)):
        got = os.path.join(prog, 'results', name.replace(code + '_', code + '_user_', 1))
        assert os.path.exists(got), got
        ref = notebook_reference_path(code, name)
        if name.endswith('таблица_кварталов.csv'):
            same_table(pd.read_csv(ref, encoding='utf-8-sig'), pd.read_csv(got, encoding='utf-8-sig'))
        else:
            kw = {'index_col': 0} if 'сравнение' in name else {}
            same_strings(pd.read_csv(ref, encoding='utf-8-sig', **kw, **STR),
                         pd.read_csv(got, encoding='utf-8-sig', **kw, **STR))
    assert os.path.exists(os.path.join(prog, 'results', f'{code}_user_для_страницы.json'))


def test_vysotsk_example_from_clean_archive(tmp_path):
    """Пример Высоцка считается из чистого архива (git archive HEAD) без докладывания файлов: слои data/vysotsk/
    в репозитории; ноутбук с ПРИМЕР = 'vysotsk' даёт исторические эталоны (запасной расчёт школ и садов);
    матрица доступности из архива подходит к слою кварталов (контрольная сумма — слой передан байт в байт)."""
    if subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True).returncode:
        pytest.skip('нет git')
    prog = zip_copy(str(tmp_path))
    for f in ['кварталы.geojson', 'здания.geojson', 'зоны.geojson', 'сервисы', 'промышленность', 'access/matrix_walk.npz']:
        assert os.path.exists(os.path.join(prog, 'data', 'vysotsk', f)), f'нет в архиве: data/vysotsk/{f}'
    nb = os.path.join(prog, 'метод_сравнения_вариантов.ipynb')
    d = json.load(open(nb, encoding='utf-8'))
    for c in d['cells']:
        c['source'] = [x.replace('ПРИМЕР = None', "ПРИМЕР = 'vysotsk'") for x in c['source']]
    json.dump(d, open(nb, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    runner = os.path.join(prog, '..', 'run_all.py')
    open(runner, 'w', encoding='utf-8').write(RUNNER + r'''
from mixuse.config import load_config
from mixuse.blocks import load_blocks
from mixuse.provision_bn import load_matrix
c = load_config('configs/vysotsk.yaml')
mx, _ = load_matrix(c, load_blocks(c)[0]['block_id'].tolist())
print('MATRIX_OK', mx.shape)
''')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(MPLBACKEND='Agg', PYTHONIOENCODING='utf-8', MIXUSE_PROVISION='fallback')
    r = subprocess.run([sys.executable, runner, nb], cwd=prog, env=env, capture_output=True, timeout=1800)
    out = r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')
    assert r.returncode == 0, out[-3000:]
    assert 'MIXUSE_FROM ' + os.path.join(prog, 'mixuse') in out.replace('/', os.sep)
    assert 'MATRIX_OK (473, 473)' in out
    ref_dir = os.path.join(ETALON_V5, 'vysotsk')
    for name in sorted(os.listdir(ref_dir)):
        got = os.path.join(prog, 'results', name)
        ref = notebook_reference_path('vysotsk', name)
        assert os.path.exists(got), got
        if name.endswith('таблица_кварталов.csv'):
            same_table(pd.read_csv(ref, encoding='utf-8-sig'), pd.read_csv(got, encoding='utf-8-sig'))
        else:
            kw = {'index_col': 0} if 'сравнение' in name else {}
            same_strings(pd.read_csv(ref, encoding='utf-8-sig', **kw, **STR),
                         pd.read_csv(got, encoding='utf-8-sig', **kw, **STR))


def _form(**extra):
    f = {'название': 'Тест', 'код': 'test', 'кварталы': {'файл': 'x.geojson'}, 'территория': [1]}
    f.update(extra)
    return f


def test_form_kit_and_exact_fields():
    """18.2: КИТ — значение с источником или строка табл. Б.1; точный расчёт — да / нет и файл OpenStreetMap."""
    from mixuse.notebook import config_from_form
    cfg = config_from_form(_form(кит={'значение': 1.4, 'источник': 'ПЗЗ'}), group_table=None, base=ROOT)
    assert cfg['params']['kit'] == 1.4 and cfg['params']['kit_source'] == 'ПЗЗ'
    assert cfg['provision'] == {'exact': 'auto', 'osm_file': None}
    cfg = config_from_form(_form(кит={'строка_СП': 'zh_multi', 'источник': 'зона О-8', 'процент_застройки_ПЗЗ': 40,
                                      'этажность_ПЗЗ': 3},
                                 точный_расчёт={'включить': True, 'файл_OSM': 'https://example.org/r.osm.pbf'}),
                           group_table=None, base=ROOT)
    P = cfg['params']
    assert (P['kit'], P['kit_sp_row'], P['pzz_build_share'], P['pzz_max_storeys']) == (None, 'zh_multi', 40, 3)
    assert cfg['param_sources']['kit_sp_row'] == 'зона О-8'
    assert cfg['provision'] == {'exact': True, 'osm_file': 'https://example.org/r.osm.pbf'}


def test_form_errors_are_plain_messages():
    """18.3: ошибки — «что не так и как исправить», в ноутбуке одной строкой «✕ …», без трассировки Python."""
    from mixuse.notebook import config_from_form
    from mixuse.blocks import DataError
    from mixuse.config import ConfigError
    from mixuse.compare import kit_value, KitError
    with pytest.raises(DataError) as e:
        config_from_form(_form(кит={'значение': 1.2, 'строка_СП': 'zh_multi'}), group_table=None, base=ROOT)
    assert 'не оба сразу' in str(e.value)
    assert e.value._render_traceback_()[0].startswith('✕ кит:')
    with pytest.raises(ConfigError) as e:
        config_from_form(_form(кит={'строка_СП': 'zh_mult'}), group_table=None, base=ROOT)
    assert 'zh_multi' in str(e.value) and e.value._render_traceback_()[0].startswith('✕')
    with pytest.raises(KitError) as e:
        kit_value({'kit': None, 'kit_sp_row': None})
    assert 'zh_low_mid — многоквартирные дома малой и средней этажности (0,8)' in str(e.value)

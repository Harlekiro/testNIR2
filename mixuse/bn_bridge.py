"""Школы и сады в вариантах: выбор способа расчёта без участия пользователя (пункт 3).

Пользователь запускает обычный Python (ноутбук или командную строку). Порядок:
  1. blocksnet установлен в этом Python и матрица есть — точный расчёт здесь же (provision_bn);
  2. рядом есть отдельное окружение с blocksnet (переменная MIXUSE_BLOCKSNET_PYTHON или
     %USERPROFILE%\\venvs\\blocksnet311\\Scripts\\python.exe — его создаёт подготовить_город.bat) — точный расчёт
     выполняется там: программа сама передаёт ему данные и забирает результат;
  3. есть сохранённый точный расчёт «как есть» (data/<код>/access/provision_walk.json) — оценка (provision_estimate);
  4. иначе — запасной расчёт 4.7.3 (prov = None).
Настройки provision.exact (анкета ноутбука «точный_расчёт», ТЗ 18.2): нет — сразу запасной расчёт; да — если данные
города для blocksnet ещё не подготовлены, программа готовит их сама (prepare: граф, матрица, расчёт «как есть»);
auto (по умолчанию в YAML) — как в пунктах 1–4. Переменная MIXUSE_PROVISION (auto, estimate, fallback) важнее настроек.
Модуль не импортирует blocksnet.
"""
import json
import os
import subprocess
import sys
import tempfile

import yaml


def blocksnet_python():
    """Путь к Python отдельного окружения с blocksnet или None."""
    cand = [os.environ.get('MIXUSE_BLOCKSNET_PYTHON'),
            os.path.join(os.path.expanduser('~'), 'venvs', 'blocksnet311', 'Scripts', 'python.exe'),
            os.path.join(os.path.expanduser('~'), 'venvs', 'blocksnet311', 'bin', 'python')]
    for p in cand:
        if p and os.path.exists(p) and os.path.abspath(p) != os.path.abspath(sys.executable):
            return p
    return None


_CFG_KEYS = ('name', 'title', 'mode', 'rules', 'blocks', 'block_id_field', 'buildings', 'services', 'service_type_field',
             'zones', 'zone_field', 'zone_rules', 'block_zone_field', 'fields',
             'territory', 'params', 'scenarios', 'service_groups', 'capacity_field', 'param_sources', 'provision',
             'zone_dictionary')


def _plain(v):
    """Значение настроек → то, что понимает YAML (кортежи → списки, числа numpy → числа)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, 'item'):
        return v.item()
    return v


def delegate(cfg, T_ids, E_ids, values, near=None, timeout=3600):
    """Точный расчёт вариантов в отдельном окружении с blocksnet (команда provision-variants). Возвращает prov."""
    py = blocksnet_python()
    if not py:
        raise RuntimeError('нет отдельного окружения с blocksnet')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as d:
        raw = {k: _plain(cfg[k]) for k in _CFG_KEYS if cfg.get(k) is not None}
        raw['params'] = {k: v for k, v in raw.get('params', {}).items()}
        # data/<код>/access/ — от корня проекта: настройки кладём в configs/ временной папки не можем, поэтому
        # передаём корень явно; пути к слоям — абсолютные
        cfg_path = os.path.join(d, 'cfg.yaml')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(raw, f, allow_unicode=True)
        job = {'territory': [int(i) for i in T_ids], 'environment': [int(i) for i in E_ids], 'near': near,
               'access_root': os.path.dirname(os.path.dirname(cfg['_path'])) if str(cfg.get('_path', '')).endswith('.yaml') else root,
               'values': {n: {'N': float(o['N']), 'd': o['d']} for n, o in values.items()}}
        job_path, out_path = os.path.join(d, 'job.json'), os.path.join(d, 'out.json')
        with open(job_path, 'w', encoding='utf-8') as f:
            json.dump(job, f, ensure_ascii=False)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
        r = subprocess.run([py, '-m', 'mixuse', 'provision-variants', cfg_path, job_path, out_path],
                           capture_output=True, timeout=timeout, env=env, cwd=root)
        if r.returncode != 0 or not os.path.exists(out_path):
            tail = (r.stdout + r.stderr).decode('utf-8', 'replace').strip().splitlines()[-3:]
            raise RuntimeError('окружение с blocksnet не посчитало: ' + ' / '.join(tail))
        return json.load(open(out_path, encoding='utf-8'))


def save_config(cfg, root=None):
    """Настройки анкеты → configs/анкета_<код>.yaml (пути абсолютные): по ним работают команды access, provision
    и подготовить_город.bat. Возвращает путь."""
    root = root or os.getcwd()
    raw = {k: _plain(cfg[k]) for k in _CFG_KEYS if cfg.get(k) is not None}
    path = os.path.join(root, 'configs', f'анкета_{cfg["name"]}.yaml')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Настройки, собранные ноутбуком из анкеты шага 0 (перезаписываются при каждом запуске).' + chr(10))
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    return path


def prepare(cfg, city, say=print):
    """Точный расчёт включён (provision.exact = да): готовит данные города для blocksnet, если их ещё нет —
    файл OpenStreetMap (скачивает по ссылке), пешеходный граф и матрица доступности (access), точный расчёт
    «как есть» (provision). Выполняется в отдельном окружении с blocksnet; один раз на город."""
    from .provision_bn import BlocksnetError, load_matrix
    from .provision_estimate import load_page_data
    try:
        if load_page_data(cfg, city) is not None:
            return
    except ValueError:
        pass                                   # данные устарели (слои изменились) — подготовить заново
    root = os.getcwd() if not str(cfg.get('_path', '')).endswith('.yaml') else os.path.dirname(os.path.dirname(cfg['_path']))
    cfg_path = cfg['_path'] if str(cfg.get('_path', '')).endswith('.yaml') else save_config(cfg, root)
    py = blocksnet_python()
    if not py:
        raise BlocksnetError('Точный расчёт школ и садов включён, но окружения с blocksnet нет. Дважды щёлкните '
                             'подготовить_город.bat в папке программы (один раз: ставит blocksnet и готовит данные города '
                             f'по настройкам {os.path.relpath(cfg_path, root)}) и выполните ноутбук снова — '
                             'или выключите точный расчёт в анкете (точный_расчёт: включить=False).')
    steps = []
    try:
        load_matrix(cfg, city['table']['block_id'], graph_type='walk')
    except (BlocksnetError, OSError, ValueError):
        osm = (cfg.get('provision') or {}).get('osm_file')
        if osm and osm.startswith(('http://', 'https://')):
            from .osm_files import download
            path = os.path.join(root, 'data', 'osm', osm.rstrip('/').split('/')[-1])
            if not os.path.exists(path):
                say(f'Школы и сады: скачиваю файл OpenStreetMap {osm} (с докачкой)…')
                download(osm, path, say=say)
            osm = path
        steps.append(('граф и матрица доступности (один раз на город; от минуты до часа)',
                      ['access', cfg_path, '--retries', '0'] + (['--osm-file', osm] if osm else [])))
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = pkg + os.pathsep + env.get('PYTHONPATH', '')
    steps.append(('точный расчёт «как есть»', ['provision', cfg_path]))
    for title, args in steps:
        say(f'Школы и сады: {title}…')
        r = subprocess.run([py, '-m', 'mixuse'] + args, capture_output=True, env=env, cwd=root)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).decode('utf-8', 'replace').strip().splitlines()[-3:]
            raise BlocksnetError(f'Подготовка данных для blocksnet не удалась ({title}): ' + ' / '.join(tail))


def school_provision(cfg, city, T_ids, det, near=None, say=print):
    """Выбирает способ расчёта школ и садов (1–4). Возвращает (prov или None, пояснение для отчёта)."""
    t = city['table']
    E_ids = t['block_id'].iloc[det['E']].astype(int).tolist()
    notes = []
    exact = (cfg.get('provision') or {}).get('exact', 'auto')
    mode = os.environ.get('MIXUSE_PROVISION') or ('fallback' if exact is False else 'auto')   # тесты эталонов — fallback
    if mode == 'fallback':
        why = 'MIXUSE_PROVISION=fallback' if os.environ.get('MIXUSE_PROVISION') else 'точный расчёт выключен в анкете'
        return None, f'Школы и сады — запасной расчёт 4.7.3 ({why}).'
    if exact is True and mode == 'auto':
        prepare(cfg, city, say=say)
    from . import provision_bn as pb
    try:                                                                     # 1. blocksnet здесь
        if mode == 'estimate':
            raise pb.BlocksnetError('выбрана оценка (MIXUSE_PROVISION=estimate)')
        pb.require_blocksnet()
        mx, _ = pb.load_matrix(cfg, t['block_id'].tolist())
        return pb.variant_provision(city, mx, T_ids, E_ids, det['values'], near=near), None
    except pb.BlocksnetError as e:
        notes.append(str(e).split('.')[0])
    if mode == 'auto' and blocksnet_python():                                # 2. отдельное окружение
        try:
            say('Школы и сады: точный расчёт в окружении с blocksnet…')
            return delegate(cfg, T_ids, E_ids, det['values'], near), None
        except Exception as e:                                               # noqa: BLE001 — сообщаем и идём дальше
            notes.append(f'окружение с blocksnet: {e}')
    from .provision_estimate import load_page_data, estimate_variants        # 3. оценка
    try:
        pdat = load_page_data(cfg, city)
    except ValueError as e:
        pdat = None
        notes.append(str(e))
    if pdat:
        prov = estimate_variants(pdat, det['T'], list(det['E']), det['values'], t['area'].to_numpy())
        return prov, ('Школы и сады — оценка по сохранённому точному расчёту blocksnet «как есть» '
                      f'(точный расчёт вариантов недоступен: {"; ".join(notes)}).')
    return None, f'Школы и сады — запасной расчёт 4.7.3 (раздел 12 не использован: {"; ".join(notes)}).'

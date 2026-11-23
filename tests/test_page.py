"""Этап 7: расчёт страницы (страница/конструктор.html) совпадает с пакетом (ТЗ, раздел 10, этап 7).

Тест вырезает из страницы блок «РАСЧЁТ» (между метками), выполняет его в Node.js на файле для страницы
(export.web_json_v5) и сверяет с compare_v5 пакета и с оценкой обеспеченности (provision_estimate) на трёх
территориях каждого набора. В файле страницы координаты, площади и жители округлены (0,01 м; 0,1 м²; 0,01 чел.),
поэтому числа сверяются с малым допуском, выводы — текстом. Если Node.js не найден — тест пропускается.
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from conftest import ROOT
from mixuse.config import load_config
from mixuse.blocks import load_layers, build_table
from mixuse.compare import compare
from mixuse.params import SCENARIO_ORIGINS, paired_scenarios
from mixuse.provision_estimate import load_page_data, estimate_variants
from mixuse import export

PAGE = os.path.join(ROOT, 'страница', 'конструктор.html')
NODE = shutil.which('node') or next((p for p in ('D:/node.exe', 'C:/Program Files/nodejs/node.exe') if os.path.exists(p)), None)

RUNNER = r'''
const fs=require('fs');
const [dataPath,argsPath]=process.argv.slice(2);
const D=JSON.parse(fs.readFileSync(dataPath,'utf8'));const A=JSON.parse(fs.readFileSync(argsPath,'utf8'));
const B=D.blocks;B.forEach(prep);
const res=[];
for(const job of A){
  const idx=new Map(B.map((b,i)=>[b.id,i]));const T=job.territory.map(id=>idx.get(id));
  const r=calc(D,B,T,{...D.params,...job.params},D.scenarios,job.kit);
  const out={};for(const [n,o] of Object.entries(r.out))out[n]={N:o.N,J:o.J,K:o.K,KA:o.KA,Kall:o.Kall,Jcity:o.Jcity,Ncity:o.Ncity,acc:o.acc,
    sP:o.sP,kP:o.kP,sT:o.sT,kT:o.kT,sNo:o.sNo,kNo:o.kNo,prov:o.prov||null,
    concl:n==='Как есть'?null:conclusion(o,r.out['Как есть'],D.params,r.city)};
  res.push({kit:r.kit,kitE:r.kitE,E:r.E.map(i=>B[i].id),accessLabel:r.accessLabel,effByFunction:r.effByFunction,out});
}
process.stdout.write(JSON.stringify(res));
'''


def calc_block():
    html = open(PAGE, encoding='utf-8').read()
    m = re.search(r'// ==== РАСЧЁТ \(начало\)[^\n]*\n(.*)// ==== РАСЧЁТ \(конец\)', html, re.S)
    assert m, 'в странице нет блока РАСЧЁТ'
    return m.group(1)


def test_calc_block_has_no_dom():
    """Блок расчёта не обращается к странице — его можно выполнить вне браузера."""
    block = calc_block()
    assert 'document' not in block and '$(' not in block


def test_city_share_comes_from_package(tmp_path):
    """Порог сравнения с городом (18.2а) на странице — только из файла, который пишет пакет (compare.CITY_SHARE)."""
    from mixuse.compare import CITY_SHARE
    block = calc_block()
    assert 'D.params.city_share' in block
    assert not re.search(r'>\s*10\b|0\.1\s*\*\s*popA|0\.15\s*\*\s*popA', block), 'порог 10 % записан на странице числом'
    for name in os.listdir(os.path.join(ROOT, 'страница', 'данные')):
        data = json.load(open(os.path.join(ROOT, 'страница', 'данные', name), encoding='utf-8'))
        assert data['params']['city_share'] == CITY_SHARE, name


def run_page(data_path, jobs, tmp_path):
    js = tmp_path / 'page_calc.js'
    js.write_text(calc_block() + RUNNER, encoding='utf-8')
    args = tmp_path / 'jobs.json'
    args.write_text(json.dumps(jobs, ensure_ascii=False), encoding='utf-8')
    r = subprocess.run([NODE, str(js), str(data_path), str(args)], capture_output=True, timeout=600)
    assert r.returncode == 0, r.stderr.decode('utf-8', 'replace')
    return json.loads(r.stdout.decode('utf-8'))


def norm_text(s):
    """Числа страницы — «0,55» и «12 345», пакета — «0.55» и «12 345»: приводим к одному виду."""
    return re.sub(r'(\d),(\d)', r'\1.\2', s.replace('\u00a0', ' '))


@pytest.mark.parametrize('code', ['vysotsk', 'gatchina_layers'])
def test_page_equals_package(code, tmp_path):
    if not NODE:
        pytest.skip('Node.js не найден')
    cfg = load_config(os.path.join(ROOT, 'configs', f'{code}.yaml'))
    if not os.path.exists(cfg['blocks']):
        pytest.skip(f'нет данных {code}')
    city = build_table(load_layers(cfg, cfg['_groups']), cfg, cfg['params'], cfg['_groups'])
    t = city['table']
    pdat = load_page_data(cfg, city)
    data = tmp_path / f'{code}.json'
    export.web_json_v5(city, cfg, str(data), pdat)
    P = cfg['params']
    kit = {'mode': 'own', 'value': P['kit']} if P.get('kit') is not None else {'mode': 'sp', 'spRow': P['kit_sp_row']}
    terrs = [list(cfg['territory']), [int(t.loc[t['pop'].idxmax(), 'block_id'])], [int(t.loc[t['cap_kinder'].idxmax(), 'block_id'])]]
    page = run_page(data, [{'territory': T, 'kit': kit} for T in terrs], tmp_path)
    for T, pg in zip(terrs, page):
        full, info, det = compare(city, T, P, cfg['scenarios'])
        prov = None
        if pdat:
            prov = estimate_variants(pdat, det['T'], list(det['E']), det['values'], t['area'].to_numpy())
            full, info, det = compare(city, T, P, cfg['scenarios'], prov=prov)
        assert sorted(pg['E']) == sorted(t['block_id'].iloc[det['E']].astype(int).tolist()), T
        assert pg['kit'] == pytest.approx(info['КИТ варианта'])
        assert pg['kitE'] == pytest.approx(info['КИТ окружения по данным (для сравнения)'], abs=0.006)
        for name, o in det['values'].items():
            q = pg['out'][name]
            for key in ('N', 'J', 'K', 'KA', 'Kall', 'Jcity', 'Ncity', 'sT', 'kT', 'sNo', 'kNo'):
                assert q[key] == pytest.approx(o[key], rel=1e-3, abs=0.01), (T, name, key)
            for g in ('К', 'ОД', 'П'):
                if o['acc']:
                    assert q['acc'][g] == pytest.approx(list(o['acc'][g]), rel=1e-3, abs=0.05), (T, name, g)
            if prov:
                for key in ('school', 'kindergarten'):
                    a, b = q['prov'][key], prov[name][key]
                    assert (a['children'], a['provided']) == (b['children'], b['provided']), (T, name, key)
                    for f in ('env_in', 'env_any'):
                        assert (a[f] is None and b[f] is None) or a[f] == pytest.approx(b[f], abs=1e-6), (T, name, key, f)
            if name != 'Как есть':
                assert norm_text(q['concl']) == det['conclusions'][name], (T, name)


def test_paired_page_single_block_matches_package(tmp_path):
    if not NODE:
        pytest.skip('Node.js не найден')
    cfg = load_config(os.path.join(ROOT, 'configs', 'vysotsk.yaml'))
    city = build_table(load_layers(cfg, cfg['_groups']), cfg, cfg['params'], cfg['_groups'])
    path = tmp_path / 'vysotsk.json'
    export.web_json_v5(city, cfg, path)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['scenarios'] == cfg['scenarios']
    assert data['paired_scenarios'] == paired_scenarios()
    assert data['scenario_origins'] == SCENARIO_ORIGINS
    assert data['blocks_crs'].endswith('CRS84')
    i = next(i for i, b in enumerate(data['blocks']) if b['id'] == 305)
    assert data['blocks'][i]['geo'] == export._geom(city['orig'][i])['coordinates']

    data['scenarios'] = data['paired_scenarios']
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    kit = {'mode': 'sp', 'spRow': cfg['params']['kit_sp_row']}
    [page] = run_page(path, [{'territory': [305], 'kit': kit}], tmp_path)
    _, _, detail = compare(city, [305], cfg['params'], paired_scenarios())
    assert list(page['out']) == ['Как есть', *paired_scenarios()]
    for name, variant in detail['values'].items():
        for metric in ('N', 'J', 'K', 'sNo', 'kNo'):
            assert page['out'][name][metric] == pytest.approx(variant[metric], rel=1e-3, abs=0.01)


DOM_STUB = r'''
const els={};
function el(id){if(!els[id])els[id]={id,value:'',innerHTML:'',textContent:'',disabled:false,style:{},files:[],
  addEventListener(){},querySelectorAll(){return [];},getContext(){return new Proxy({},{get:()=>()=>{}});},
  getBoundingClientRect(){return {width:800,height:560,left:0,top:0};}};return els[id];}
global.document={getElementById:el};global.window={addEventListener(){},devicePixelRatio:1};
'''


@pytest.mark.parametrize('code', ['vysotsk', 'gatchina_layers'])
def test_page_renders(code, tmp_path):
    """Вся страница (не только расчёт) выполняется без ошибок: загрузка файла, расчёт, краткая и подробная таблицы."""
    if not NODE:
        pytest.skip('Node.js не найден')
    data = os.path.join(ROOT, 'страница', 'данные', f'{code}.json')
    if not os.path.exists(data):
        pytest.skip('нет файла примера')
    html = open(PAGE, encoding='utf-8').read()
    script = re.search(r'<script>(.*)</script>', html, re.S).group(1)
    js = tmp_path / 'page_full.js'
    js.write_text(DOM_STUB + script + r'''
const D0=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
use(D0);
const res=document.getElementById('res').innerHTML,full=document.getElementById('resFull').innerHTML;
document.getElementById('scenarioMode').value='paired';
document.getElementById('scenarioMode').onchange();
const joint=document.getElementById('res').innerHTML;
const access=document.getElementById('accessMode'),threshold=document.getElementById('walkMinutes');
access.value='walk';access.onchange();
const network=document.getElementById('res').innerHTML;
threshold.value=21;threshold.onchange();
const rejected=last===null&&document.getElementById('geojson').disabled;
delete D0.walk_access;use(D0);
const fallback=access.value==='geometry'&&document.getElementById('walkOption').disabled;
process.stdout.write(JSON.stringify({res,full,kit:document.getElementById('kitSrc').textContent,
  joint,network,rejected,fallback,info:document.getElementById('info').innerHTML,warn:document.getElementById('warn').innerHTML}));
''', encoding='utf-8')
    r = subprocess.run([NODE, str(js), data], capture_output=True, timeout=600)
    assert r.returncode == 0, r.stderr.decode('utf-8', 'replace')
    out = json.loads(r.stdout.decode('utf-8'))
    assert 'Вывод' in out['res'] and 'Жители на территории' in out['res'] and 'Ж+К+ОД' in out['res']
    assert '[эмп.]' not in out['res'] and all(name in out['joint'] for name in paired_scenarios())
    assert 'пешком ≤ 10 мин' in out['network'] and out['rejected'] and out['fallback']
    assert 'обеспеченность окружения в пределах доступности' in out['full']
    assert 'СП 42.13330.2016' in out['kit']                       # Высоцк и Гатчина: КИТ по СП (kit_sp_row)
    assert 'раз плотнее' in out['info'] or 'раз реже' in out['info']


def test_closed_facility_warning(tmp_path):
    """Предупреждение «закрываются учреждения территории» — в пакете и на странице одинаково (Гатчина, квартал 147:
    крупнейший сад; 40 % дошкольников окружения ходили туда)."""
    if not NODE:
        pytest.skip('Node.js не найден')
    cfg = load_config(os.path.join(ROOT, 'configs', 'gatchina_layers.yaml'))
    if not os.path.exists(cfg['blocks']):
        pytest.skip('нет данных Гатчины')
    city = build_table(load_layers(cfg, cfg['_groups']), cfg, cfg['params'], cfg['_groups'])
    pdat = load_page_data(cfg, city)
    if not pdat:
        pytest.skip('нет точного расчёта «как есть»')
    t = city['table']
    T = [int(t.loc[t['cap_kinder'].idxmax(), 'block_id'])]
    _, _, det = compare(city, T, cfg['params'], cfg['scenarios'])
    prov = estimate_variants(pdat, det['T'], list(det['E']), det['values'], t['area'].to_numpy())
    _, info, _ = compare(city, T, cfg['params'], cfg['scenarios'], prov=prov)
    pkg = [w for w in info['предупреждения'] if 'закрываются учреждения' in w]
    assert len(pkg) == 1 and 'дошкольников' in pkg[0]
    T0 = list(cfg['territory'])
    _, _, det0 = compare(city, T0, cfg['params'], cfg['scenarios'])
    prov0 = estimate_variants(pdat, det0['T'], list(det0['E']), det0['values'], t['area'].to_numpy())
    _, info2, _ = compare(city, T0, cfg['params'], cfg['scenarios'], prov=prov0)
    assert not any('закрываются учреждения' in w for w in info2['предупреждения'])     # территория из настроек — нет
    data = tmp_path / 'g.json'
    export.web_json_v5(city, cfg, str(data), pdat)
    html = open(PAGE, encoding='utf-8').read()
    block = re.search(r'// ==== РАСЧЁТ \(начало\)[^\n]*\n(.*)// ==== РАСЧЁТ \(конец\)', html, re.S).group(1)
    js = tmp_path / 'w.js'
    js.write_text(block + r'''
const D=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));const B=D.blocks;B.forEach(prep);
const idx=new Map(B.map((b,i)=>[b.id,i]));
const r=calc(D,B,[idx.get(+process.argv[3])],D.params,D.scenarios,{mode:'sp',spRow:D.params.kit_sp_row});
process.stdout.write(JSON.stringify(r.warns));''', encoding='utf-8')
    out = subprocess.run([NODE, str(js), str(data), str(T[0])], capture_output=True, timeout=300)
    assert out.returncode == 0, out.stderr.decode('utf-8', 'replace')
    page = [w for w in json.loads(out.stdout.decode('utf-8')) if 'закрываются учреждения' in w]
    assert len(page) == 1 and norm_text(page[0]) == pkg[0]

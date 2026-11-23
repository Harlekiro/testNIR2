"""Разреженное представление существующей матрицы времени для доступа к К и ОД."""
from math import isfinite
from zipfile import BadZipFile

import numpy as np

from .errors import UserMessage

DEFAULT_WALK_MINUTES = 10
DEFAULT_PREPARED_MINUTES = 20


class WalkAccessError(UserMessage, ValueError):
    pass


def check_threshold(minutes, maximum):
    if not (isfinite(minutes) and isfinite(maximum) and 0 < minutes <= maximum):
        raise WalkAccessError('Порог ходьбы должен быть положительным и не превышать подготовленный максимум.')


def load_walk_access(cfg, block_ids, minutes=DEFAULT_WALK_MINUTES, maximum=None, keep=None):
    """Проверить сохранённую матрицу и оставить направленные пары в пределах максимума.

    Строки и номера назначения относятся к порядку block_ids в результате, включая при обрезке.
    Каждый ряд — плоский список [индекс, минуты, индекс, минуты, …].
    """
    from .provision_bn import BlocksnetError, load_matrix

    maximum = minutes if maximum is None else maximum
    check_threshold(minutes, maximum)
    ids = [int(i) for i in block_ids]
    try:
        matrix, meta = load_matrix(cfg, ids, graph_type='walk')
        if (meta.get('граф') != 'walk' or meta.get('единицы') != 'минуты'
                or not meta.get('sha256 слоя кварталов')):
            raise WalkAccessError('Для матрицы не подтверждены пешеходный граф, минуты и контрольная сумма кварталов.')
    except (BlocksnetError, OSError, ValueError, BadZipFile, KeyError, AttributeError) as exc:
        return None, str(exc)

    keep = np.arange(len(ids)) if keep is None else np.asarray(keep, dtype=int)
    values = matrix.to_numpy(copy=False)
    rows = []
    for i in keep:
        # Сравнение в float64: порог пользователя не должен округляться до float16 матрицы.
        time = values[i, keep].astype(float)
        destinations = np.flatnonzero(np.isfinite(time) & (time >= 0) & (time <= maximum))
        row = []
        for j in destinations:
            row.extend((int(j), float(time[j])))
        rows.append(row)
    return {'method': 'walk', 'units': 'minutes', 'source': 'OpenStreetMap', 'graph': 'walk',
            'graph_source': meta.get('источник графа', 'OpenStreetMap'),
            'threshold_minutes': minutes, 'max_minutes': maximum,
            'block_ids': [ids[i] for i in keep], 'rows': rows}, None


def walk_neighbors(access, block_ids, origins):
    """Индексы достижимых назначений для заданных строк проживания."""
    ids = list(block_ids)
    rows = access['rows']
    if ((access['method'], access['graph'], access['units']) != ('walk', 'walk', 'minutes')
            or access['block_ids'] != ids or len(rows) != len(ids)):
        raise WalkAccessError('Пешеходные связи не соответствуют кварталам расчёта или заданы не в минутах.')
    minutes = access['threshold_minutes']
    check_threshold(minutes, access['max_minutes'])
    return {i: np.asarray([j for j, time in zip(rows[i][::2], rows[i][1::2]) if time <= minutes], dtype=int)
            for i in origins}

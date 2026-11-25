"""Рабочие места в сценариях по кварталам-аналогам и вместимости помещений."""
import numpy as np


JOB_GROUPS = ('К', 'ОД', 'П')


def employment_stats(areas, jobs):
    """По каждой функции: число аналогов, их площадь земли, рабочие места и категория А."""
    jobs = np.asarray(jobs)
    areas = np.asarray(areas)[:, None]
    peers = (areas > 0) & (jobs[:, :, 0] > 0)
    return np.column_stack((peers.sum(axis=0), np.where(peers, areas, 0).sum(axis=0),
                            np.where(peers[:, :, None], jobs, 0).sum(axis=0)))


def context_rates(areas, jobs, territory, environment):
    """Удельная занятость на м² земли. Территория исключена из обоих наборов аналогов."""
    local = employment_stats(areas[environment], jobs[environment])
    elsewhere = np.ones(len(areas), dtype=bool)
    elsewhere[territory] = elsewhere[environment] = False
    reference = employment_stats(areas[elsewhere], jobs[elsewhere])
    rates = {}
    for group, (count, area, total, known), (ref_count, ref_area, ref_total, ref_known) in zip(
            JOB_GROUPS, local, reference):
        prior = np.array([ref_total, ref_known]) / ref_area if ref_count else np.zeros(2)
        if count:
            # Один аналог средней местной площади сглаживает редкие наблюдения.
            weight = area / count if ref_count else 0
            density = (np.array([total, known]) + weight * prior) / (area + weight)
        else:
            density = prior
        rates[group] = (*density, int(count), int(ref_count))
    return rates


def scenario_jobs(area, kit, shares, params, rates):
    """Оценка рабочих мест, её компонент А и вместимость проектной площади."""
    jobs, known, capacity, _ = scenario_employment(area, kit, shares, params, rates)
    return jobs, known, capacity


def scenario_employment(area, kit, shares, params, rates):
    """Возвращает итог занятости и вклад каждой функции для слоя кварталов.

    Доли поэтажной площади взвешивают удельную занятость типовых кварталов.
    Вклад каждой функции ограничен её собственной площадью и нормой м² на место.
    rates=None воспроизводит расчёт вместимости прежней методики.
    """
    if rates is None:
        floor_area = area * kit
        floor_rates = {group: shares[group] / params['a_' + group] for group in JOB_GROUPS}
        capacity = floor_area * sum(floor_rates.values())
        return capacity, capacity, capacity, {g: floor_area * rate for g, rate in floor_rates.items()}
    jobs = known = capacity = 0.0
    by_function = {}
    for group in JOB_GROUPS:
        if not shares[group]:
            by_function[group] = 0.0
            continue
        equivalent_area = area * shares[group]
        limit = equivalent_area * kit / params['a_' + group]
        capacity += limit
        density, known_density, _, _ = rates[group]
        contribution = min(limit, equivalent_area * density)
        by_function[group] = contribution
        jobs += contribution
        if density > 0:
            known += contribution * known_density / density
    return jobs, known, capacity, by_function

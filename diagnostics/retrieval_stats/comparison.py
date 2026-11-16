"""Pair saved scopes at identical query/seed/LR; no model or cache execution."""

from diagnostics.common import require


MATCHED = ('dataset', 'reference', 'validation_manifest', 'fresh_queries', 'collection', 'collection_policy',
           'split', 'queries', 'full_queries', 'limit', 'seeds', 'methods', 'learning_rates', 'ks',
           'schedule', 'horizon', 'device', 'cpu_threads', 'support_size')
PAIR_KEY = ('dataset', 'query_id', 'seed', 'method', 'learning_rate', 'position', 'comparison', 'selection', 'k')


def validate_pair(current, baseline):
    require(current['scope'] == 'predictor' and baseline['scope'] == 'last-block',
            'Scope comparison requires predictor versus a completed last-block baseline')
    different = [name for name in MATCHED if current[name] != baseline[name]]
    require(not different, f'Mismatched scope-comparison protocol fields: {different}')


def validate_runtime(current, baseline):
    # Both workers verify the numerical source against the same pinned report.
    # Equivalent verified archives can live at different directory paths.
    # Scope descriptions/counts differ intentionally; cached/model identities do not.
    keys = ('model', 'memory', 'query_cache', 'retrieval', 'memory_windows', 'device', 'torch', 'cuvs')
    different = [name for name in keys if current.get(name) != baseline.get(name)]
    require(not different, f'Mismatched scope-comparison runtime fields: {different}')
    require(current['scope']['scope'] == 'predictor' and baseline['scope']['scope'] == 'last-block',
            'Recorded trainable scopes disagree with the comparison')


def paired_rows(current, baseline):
    """Pointwise predictor-minus-last-block effects; cluster episodes later."""
    def indexed(rows):
        result = {}
        for row in rows:
            key = tuple(row[name] for name in PAIR_KEY)
            require(key not in result, 'Repeated query/seed/LR/position statistic')
            result[key] = row
        return result

    now, old = indexed(current), indexed(baseline)
    require(now.keys() == old.keys(), 'Scope comparison has unmatched query/seed/LR/position records')
    result = []
    for key, row in now.items():
        previous = old[key]
        require(row['episode_id'] == previous['episode_id'], 'Paired query belongs to another episode')
        if row['position'] == 0 and row['comparison'] == 'forward_reverse_before':
            for field in ('before_size', 'after_size', 'intersection', 'jaccard', 'episode_intersection'):
                require(row[field] == previous[field], 'Pre-adaptation original-context retrieval differs across scopes')
        if row['comparison'] == 'reverse_after_update':
            metrics = ('new_fraction', 'intersection', 'jaccard', 'set_changed', 'prediction_correction_mse')
        elif row['comparison'] == 'fr_support_after_update':
            metrics = ('added', 'weight_total_variation')
        elif row['comparison'] == 'reverse_between_positions':
            metrics = ('new_fraction',)
        else:
            continue
        values = {field: row[field]-previous[field] for field in metrics
                  if row[field] is not None and previous[field] is not None}
        result.append({**dict(zip(PAIR_KEY, key)), 'scope': 'predictor-minus-last-block',
                       'episode_id': row['episode_id'], **values})
    return result


def format_comparison(summary, differences, runtimes):
    lines = ['# Whole predictor versus final-block reverse-pool response', '',
        'Only the predictor condition is newly measured. The last-block records are reused read-only. '
        'Both use reverse-only B32 temporal-U1, the same original checkpoint, queries, actions, LR grid '
        'and seeds. The action/visual encoders and source retrieval index stay frozen.', '',
        'At position0 both conditions have the same observed context and pre-update support. Later '
        'positions follow their own adapted autoregressive trajectories; their comparison describes '
        'the two complete procedures and does not hold cross-scope context fixed.', '',
        '| Dataset | Update method | LR | Prefix k | Positions | Last-block R replaced | Predictor R replaced | Predictor minus last-block, pp [95% CI] |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    lookup = {tuple(row[name] for name in ('dataset', 'scope', 'method', 'learning_rate', 'k', 'position', 'metric')): row
              for row in summary if row['seed'] == 'all' and row['comparison'] == 'reverse_after_update'
              and row['selection'] == 'diverse'}
    for row in differences:
        if (row['seed'] != 'all' or row['comparison'] != 'reverse_after_update' or row['selection'] != 'diverse'
                or row['metric'] != 'new_fraction' or row['position'] not in (0, 'all')):
            continue
        key = (row['dataset'], row['method'], row['learning_rate'], row['k'], row['position'], 'new_fraction')
        base = lookup[(key[0], 'last-block', *key[1:])]['mean']
        whole = lookup[(key[0], 'predictor', *key[1:])]['mean']
        interval = ('unavailable' if row['ci95_low'] is None else
                    f"{row['ci95_low']*100:+.2f}, {row['ci95_high']*100:+.2f}")
        positions = 'initial' if row['position'] == 0 else 'all five'
        lines.append(f"| {row['dataset']} | {row['method']} | {row['learning_rate']:g} | {row['k']} | {positions} | "
                     f"{base*100:.2f}% | {whole*100:.2f}% | {row['mean']*100:+.2f} [{interval}] |")
    lines += ['', 'Intervals bootstrap paired differences by episode, carrying all windows/seeds/positions '
              'together. Positive differences mean more sample replacement. Both the original-context '
              'and all-position effects are in `scope-comparison.csv.gz`.', '',
              '| Dataset | Final-block parameters | Predictor parameters |', '|---|---:|---:|']
    for dataset, runtime in runtimes.items():
        lines.append(f"| {dataset} | {runtime['last-block']['scope']['trainable_parameters']:,} | "
                     f"{runtime['predictor']['scope']['trainable_parameters']:,} |")
    lines += ['', '| Dataset | Update method | LR | Prefix k | Initial prediction correction MSE, last-block | Initial prediction correction MSE, predictor |',
              '|---|---|---:|---:|---:|---:|']
    for row in summary:
        if (row['scope'] != 'predictor' or row['seed'] != 'all' or row['position'] != 0
                or row['comparison'] != 'reverse_after_update' or row['selection'] != 'diverse'
                or row['metric'] != 'prediction_correction_mse'):
            continue
        key = (row['dataset'], row['method'], row['learning_rate'], row['k'], 0, 'prediction_correction_mse')
        base = lookup[(key[0], 'last-block', *key[1:])]['mean']
        lines.append(f"| {row['dataset']} | {row['method']} | {row['learning_rate']:g} | {row['k']} | "
                     f"{base:.5g} | {row['mean']:.5g} |")
    lines += ['', 'The existing `prediction_correction_mse` statistic measures functional movement on '
              'the query, not error against its unseen future. It is also paired across scopes in '
              '`scope-comparison.csv.gz`. Larger churn or correction does not establish better prediction '
              'or planning. Shared LR and clipping thresholds do not imply equal effective update size '
              'across different parameter scopes.']
    return '\n'.join(lines)+'\n'

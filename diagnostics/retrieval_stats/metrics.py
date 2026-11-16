"""Small set/weight statistics, independent of Torch and the experiment core."""

import math


def compare(before, after, before_weights=None, after_weights=None):
    """Exact transition identity; index order is retained for reorder statistics."""
    before, after = list(map(int, before)), list(map(int, after))
    left, right = set(before), set(after)
    if len(left) != len(before) or len(right) != len(after):
        raise ValueError('A retrieval list contains repeated transition indices')
    common, union = left & right, left | right
    first, second = {v: i for i, v in enumerate(before)}, {v: i for i, v in enumerate(after)}
    result = {'before_size': len(left), 'after_size': len(right), 'intersection': len(common),
              'union': len(union), 'added': len(right-left), 'removed': len(left-right),
              'overlap_fraction': len(common)/min(len(left), len(right)) if left and right else None,
              'retained_fraction': len(common)/len(left) if left else None,
              'new_fraction': len(right-left)/len(right) if right else None,
              'jaccard': len(common)/len(union) if union else None,
              'set_changed': int(left != right), 'order_changed': int(before != after),
              'retained_rank_displacement': (sum(abs(first[v]-second[v]) for v in common)/len(common)
                                              if common else None), 'weight_total_variation': None}
    if (before_weights is None) != (after_weights is None):
        raise ValueError('Supply both normalized weight vectors or neither')
    if before_weights is not None:
        for indices, weights in ((before, before_weights), (after, after_weights)):
            if (len(indices) != len(weights) or any(not math.isfinite(w) or w < 0 for w in weights)
                    or not math.isclose(sum(weights), 1., rel_tol=0, abs_tol=1e-6)):
                raise ValueError('Support weights must be finite, nonnegative and sum to one')
        w0, w1 = dict(zip(before, before_weights)), dict(zip(after, after_weights))
        result['weight_total_variation'] = .5*sum(abs(w0.get(v, 0)-w1.get(v, 0)) for v in union)
    return result


def episode_overlap(before, after, episodes):
    left = {int(episodes[int(v)]) for v in before}
    right = {int(episodes[int(v)]) for v in after}
    return {'before_episodes': len(left), 'after_episodes': len(right),
            'episode_intersection': len(left & right),
            'episode_jaccard': len(left & right)/len(left | right) if left or right else None}

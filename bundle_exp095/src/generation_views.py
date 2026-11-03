"""Eight spatial views, one color permutation; preserve every output. MIT-0."""
def batches_for_outputs(outputs, tokens_by_key):
    """Group surviving views by actual prefix length, with at most four per batch."""
    batches = []
    for keys in outputs.values():
        by_length = {}
        for key in keys:
            by_length.setdefault(len(tokens_by_key[key]), []).append(key)
        for group in by_length.values():
            batches.extend(group[i:i+4] for i in range(0, len(group), 4))
    return batches

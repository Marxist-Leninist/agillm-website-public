"""Pure presentation metadata; never contacts or mutates the training host."""
from datetime import datetime, timezone
import math

REVISION = '32b-provenance-v1-20261013'


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    if valid_number(value):
        return value if value >= 0 else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            return None
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return None


def enrich_metadata(result, now):
    """Add checkpoint-specific freshness without changing existing measured values."""
    training = result.setdefault('training', {})
    saved = (result.get('saved') or {}).get('ckpt') or {}
    serving = result.get('serving') or {}
    current_step, saved_step, served_step = training.get('step'), saved.get('step'), serving.get('step')
    eval_step, nats = training.get('heldout_step'), training.get('heldout_nats')
    evaluated_at = timestamp(training.get('heldout_at'))
    present = valid_number(eval_step) and valid_number(nats) and nats >= 0
    saved_match = present and valid_number(saved_step) and eval_step == saved_step
    served_match = present and valid_number(served_step) and eval_step == served_step
    historical = present and valid_number(saved_step) and eval_step != saved_step
    reason = ('missing' if not present else 'checkpoint_mismatch' if historical else
              'timestamp_missing' if evaluated_at is None else 'saved_checkpoint_unknown' if not valid_number(saved_step) else None)
    status = ('unavailable' if not present else 'historical_checkpoint' if historical else
              'latest_saved_checkpoint' if saved_match else 'checkpoint_unverified')
    evaluation = {
        'available': present,
        'status': status,
        'evaluated_checkpoint_step': eval_step if present else None,
        'evaluated_at': training.get('heldout_at'),
        'evaluated_at_epoch': evaluated_at,
        'age_s': max(0.0, now - evaluated_at) if evaluated_at is not None else None,
        'matches_latest_saved_checkpoint': saved_match,
        'matches_served_checkpoint': served_match,
        'stale': reason is not None,
        'stale_reason': reason,
        'nll_nats': nats if present else None,
        'protocol': training.get('heldout_protocol'),
        'protocol_may_be_truncated': len(str(training.get('heldout_protocol') or '')) >= 80,
        'metric_scope': 'frozen held-out next-token evaluation; preserve its own checkpoint and protocol',
        'checkpoint_sha256': None,
        'checkpoint_sha256_verified': False,
        'warning': ('Historical evaluation; not a quality score for the latest trained, saved, or served model.'
                    if historical and not served_match else
                    'Historical training checkpoint evaluation; see checkpoint-specific serving match.'
                    if historical else None),
    }
    training.update(
        sampled_training_tokens_seen=training.get('fresh_tokens_seen'),
        sampled_training_tokens_per_second=training.get('fresh_tokens_per_second'),
        loss_target_exposures_seen=training.get('objective_tokens_seen'),
        loss_target_exposures_per_second=training.get('objective_tokens_per_second_window'),
        unique_corpus_tokens_seen=None,
        heldout_age_s=evaluation['age_s'],
        heldout_stale=evaluation['stale'],
        heldout_status=status,
        heldout_matches_saved_checkpoint=saved_match,
        heldout_matches_served_checkpoint=served_match,
        runtime_revision_reported=bool(training.get('runtime_revision')),
        runtime_revision_verified=False,
        objective_coverage_verified=False,
        mode_window_definition='Observed losses in a bounded recent log window; absence does not prove an objective is disabled.',
        fresh_tokens_definition='Sampled input-token occurrences when a new batch is loaded; corpus sampling can repeat tokens. Not unique corpus coverage.',
        objective_tokens_definition='Loss-target exposures summed over sublayer updates, including repeated use of a batch. Not fresh data.',
    )
    result['evaluation'] = evaluation
    result['lineage'] = {
        'trained_step': current_step,
        'saved_step': saved_step,
        'served_step': served_step,
        'evaluated_step': eval_step if present else None,
        'saved_sha256': saved.get('sha256'),
        'saved_at': saved.get('updated_at'),
        'saved_age_s': max(0.0, now - saved['updated_at']) if valid_number(saved.get('updated_at')) else None,
        'training_steps_since_saved': current_step - saved_step if valid_number(current_step) and valid_number(saved_step) else None,
        'saved_steps_ahead_of_served': saved_step - served_step if valid_number(saved_step) and valid_number(served_step) else None,
        'serving_checkpoint_sha256_verified': False,
    }
    result['freshness'] = {
        'training': 'stale' if result.get('stale', True) else 'fresh',
        'training_age_s': result.get('training_age_s'),
        'serving': serving.get('status', 'unknown'),
        'evaluation': status,
        'evaluation_age_s': evaluation['age_s'],
        'meaning': 'Fresh training telemetry does not make a historical evaluation fresh.',
    }
    result['metadata_revision'] = REVISION
    return result

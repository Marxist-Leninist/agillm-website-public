/* AGILLM_CHECKPOINT_RETENTION_V2: checkpoint metadata only; no model state. */
(function(root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.AGILLMCheckpointRetention = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  const models = ['standard', 'rpv16', 'bintf4b'];
  function stepNumber(value) {
    if (typeof value !== 'number' && !(typeof value === 'string' && /^\d+$/.test(value))) return null;
    const n = Number(value);
    return Number.isSafeInteger(n) && n >= 0 ? n : null;
  }
  function create(options) {
    const parseTime = options.parseTime;
    const storage = options.storage;
    const now = options.now || Date.now;
    const ttl = 7 * 86400000;
    const storageKey = 'agillm_checkpoint_times_v2';
    const records = Object.create(null), confirmed = Object.create(null);
    function valid(record) {
      return record && stepNumber(record.step) !== null && Number.isFinite(record.seenAt) &&
        record.seenAt <= now() + 60000 && now() - record.seenAt <= ttl &&
        typeof record.scope === 'string' && record.scope.length <= 240 &&
        typeof record.identity === 'string' && record.identity.length <= 128 &&
        (record.at === null || parseTime(record.at) !== null);
    }
    try {
      const data = JSON.parse(storage?.getItem(storageKey) || 'null');
      if (data?.version === 2) for (const model of models) {
        if (valid(data.records?.[model])) records[model] = data.records[model];
      }
    } catch (_) { /* Disabled or corrupt browser storage must not break the page. */ }
    function persist() {
      try { storage?.setItem(storageKey, JSON.stringify({version:2, records})); } catch (_) {}
    }
    function observe(model, event, scope = '') {
      if (!models.includes(model)) return;
      scope = typeof scope === 'string' ? scope.slice(0,240) : '';
      const step = stepNumber(event?.step);
      // Retention fix 20261008: a degraded feed block (missing model_id/lineage/
      // instance_id) must never wipe last-known data; only a genuine observation
      // with a complete scope retires a previous lineage record.
      const scopeParts = scope.split('|');
      const scopeComplete = scopeParts.length === 3 && scopeParts.every(p => p !== '');
      if (records[model] && (!valid(records[model]) || (step !== null && scopeComplete && records[model].scope && scope !== records[model].scope))) {
        delete records[model];
      }
      confirmed[model] = false;
      if (step === null) { persist(); return; }
      const at = parseTime(event?.at);
      const identity = typeof event?.identity === 'string' ? event.identity.slice(0,128) : '';
      const previous = records[model];
      const same = previous && previous.step === step &&
        (!identity || !previous.identity || identity === previous.identity);
      if (at !== null) {
        // A restored checkpoint can have a LOWER step. Never impose max-step selection.
        records[model] = {step, at, seenAt:now(), scope, identity};
        confirmed[model] = true;
      } else if (!same) {
        // Never attach a previous checkpoint's timestamp to a different step or hash.
        records[model] = {step, at:null, seenAt:now(), scope, identity};
      }
      persist();
    }
    function unavailable(model) { if (models.includes(model)) confirmed[model] = false; }
    function view(model) {
      const record = records[model];
      if (!valid(record)) return {saved:null, savedAt:null, savedLastKnown:false};
      return {saved:record.step, savedAt:record.at, savedLastKnown:confirmed[model] !== true};
    }
    return Object.freeze({observe, unavailable, view});
  }
  return Object.freeze({create, stepNumber});
});

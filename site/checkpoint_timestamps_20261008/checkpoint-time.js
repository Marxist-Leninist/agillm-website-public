/* Checkpoint timestamps. Callers must pass checkpoint metadata, not poll times. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AGILLMCheckpointTime = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const earliest = Date.UTC(2000, 0, 1);
  const latest = Date.UTC(2100, 0, 1);
  const absoluteFormatter = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Europe/London', year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
    timeZoneName: 'short'
  });

  function parse(value) {
    let ms = NaN;
    if (typeof value === 'number') {
      ms = value < 1e11 ? value * 1000 : value;
    } else if (typeof value === 'string' && value.trim()) {
      const text = value.trim();
      if (/^\d+(?:\.\d+)?$/.test(text)) {
        const n = Number(text);
        ms = n < 1e11 ? n * 1000 : n;
      } else if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})$/i.test(text)) {
        const year = Number(text.slice(0, 4));
        const month = Number(text.slice(5, 7));
        const day = Number(text.slice(8, 10));
        if (month >= 1 && month <= 12 && day >= 1 &&
            day <= new Date(Date.UTC(year, month, 0)).getUTCDate()) ms = Date.parse(text);
      }
    }
    return Number.isFinite(ms) && ms >= earliest && ms < latest ? ms : null;
  }

  function age(ms, now) {
    const delta = now - ms;
    if (!Number.isFinite(delta)) return '';
    if (delta < -60000) return 'timestamp is ahead of this device';
    const minutes = Math.floor(Math.max(0, delta) / 60000);
    if (minutes === 0) return 'less than a minute ago';
    if (minutes < 60) return minutes + ' min ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + ' h' + (minutes % 60 ? ' ' + minutes % 60 + ' min' : '') + ' ago';
    const days = Math.floor(hours / 24);
    return days + (days === 1 ? ' day' : ' days') + ' ago';
  }

  function format(value, now = Date.now()) {
    const ms = parse(value);
    if (ms === null) return null;
    return {
      ms,
      iso: new Date(ms).toISOString(),
      absolute: absoluteFormatter.format(ms),
      relative: age(ms, now)
    };
  }

  function render(target, value, options = {}) {
    if (!target || !target.ownerDocument) return false;
    const doc = target.ownerDocument;
    const result = format(value, options.now === undefined ? Date.now() : options.now);
    const label = options.label || 'Last updated';
    target.classList.add('checkpoint-last-updated');
    if (!result) {
      target.textContent = label + ': time unavailable';
      target.removeAttribute('title');
      return false;
    }
    const time = doc.createElement('time');
    time.dateTime = result.iso;
    time.textContent = result.absolute;
    time.title = result.iso + (options.source ? ' · ' + options.source : '');
    const relative = doc.createElement('span');
    relative.className = 'checkpoint-age';
    relative.textContent = result.relative ? ' (' + result.relative + ')' : '';
    target.replaceChildren(doc.createTextNode(label + ': '), time, relative);
    target.title = options.source || 'Checkpoint timestamp; not the page refresh time.';
    return true;
  }

  return Object.freeze({ parse, age, format, render });
});

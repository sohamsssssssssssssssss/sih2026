/* SatQuery API boundary for the imported reference frontend. */
(function (global) {
  'use strict';

  const DEFAULT_API_BASE = 'http://127.0.0.1:8000';
  const DEFAULT_TIMEOUT_MS = 10000;
  const EXECUTION_MODES = new Set(['live', 'cached_result']);

  class SatQueryApiError extends Error {
    constructor(message, options) {
      super(message);
      this.name = 'SatQueryApiError';
      this.kind = options.kind;
      this.status = options.status ?? null;
    }
  }

  function apiBase() {
    const configured = typeof global.SATQUERY_API_BASE === 'string'
      ? global.SATQUERY_API_BASE.trim()
      : '';
    const candidate = configured || DEFAULT_API_BASE;
    let parsed;
    try {
      parsed = new URL(candidate);
    } catch {
      throw new SatQueryApiError('The SatQuery API address is invalid.', { kind: 'configuration' });
    }
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      throw new SatQueryApiError('The SatQuery API address must use HTTP or HTTPS.', { kind: 'configuration' });
    }
    return parsed.href.replace(/\/$/, '');
  }

  function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function requireRecord(value, label) {
    if (!isRecord(value)) {
      throw new SatQueryApiError(`The API returned an invalid ${label} response.`, { kind: 'invalid_response' });
    }
    return value;
  }

  function requireString(record, field, label) {
    if (typeof record[field] !== 'string' || !record[field].trim()) {
      throw new SatQueryApiError(`The API returned an invalid ${label} response.`, { kind: 'invalid_response' });
    }
  }

  function cleanHttpMessage(status) {
    if (status === 404) return 'The requested SatQuery resource is unavailable.';
    if (status === 422) return 'SatQuery rejected the request. Check the submitted inputs.';
    if (status === 503) return 'The SatQuery service or a required local artifact is unavailable.';
    if (status >= 500) return 'The SatQuery service could not complete the request.';
    return 'The SatQuery API rejected the request.';
  }

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : DEFAULT_TIMEOUT_MS;
    const timeout = global.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await global.fetch(`${apiBase()}${path}`, {
        method: options.method || 'GET',
        headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
        cache: 'no-store'
      });
      if (!response.ok) {
        throw new SatQueryApiError(cleanHttpMessage(response.status), {
          kind: 'backend_rejection',
          status: response.status
        });
      }
      try {
        return await response.json();
      } catch {
        throw new SatQueryApiError('The SatQuery API returned an unreadable response.', {
          kind: 'invalid_response',
          status: response.status
        });
      }
    } catch (error) {
      if (error instanceof SatQueryApiError) throw error;
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new SatQueryApiError('The SatQuery API request timed out.', { kind: 'timeout' });
      }
      throw new SatQueryApiError('The SatQuery API is unavailable.', { kind: 'network' });
    } finally {
      global.clearTimeout(timeout);
    }
  }

  function validateHealth(value) {
    const data = requireRecord(value, 'health');
    requireString(data, 'status', 'health');
    requireString(data, 'mode', 'health');
    return { status: data.status, mode: data.mode };
  }

  function validateAnalysis(value) {
    const data = requireRecord(value, 'analysis');
    requireString(data, 'answer', 'analysis');
    requireString(data, 'execution_mode', 'analysis');
    requireString(data, 'notice', 'analysis');
    if (!EXECUTION_MODES.has(data.execution_mode)) {
      throw new SatQueryApiError('The API returned an unknown execution mode.', { kind: 'invalid_response' });
    }
    const model = requireRecord(data.model, 'analysis');
    requireString(model, 'name', 'analysis');
    requireString(model, 'version', 'analysis');
    const trace = requireRecord(data.trace, 'analysis');
    if (data.results_artifact !== null && typeof data.results_artifact !== 'string') {
      throw new SatQueryApiError('The API returned an invalid analysis response.', { kind: 'invalid_response' });
    }
    return { ...data, model: { name: model.name, version: model.version }, trace };
  }

  function validateResolution(value) {
    const data = requireRecord(value, 'resolution');
    requireString(data, 'model', 'resolution');
    requireString(data, 'timestamp', 'resolution');
    if (!Number.isFinite(data.n_samples) || !isRecord(data.per_rung) || !Array.isArray(data.degenerate_rungs)) {
      throw new SatQueryApiError('The API returned an invalid resolution response.', { kind: 'invalid_response' });
    }
    return data;
  }

  function validateSar(value) {
    const data = requireRecord(value, 'SAR');
    requireString(data, 'scene', 'SAR');
    requireString(data, 'title', 'SAR');
    requireString(data, 'annotation', 'SAR');
    if (data.human_validation !== true || typeof data.render_available !== 'boolean' || !isRecord(data.summaries)) {
      throw new SatQueryApiError('The API returned an invalid SAR response.', { kind: 'invalid_response' });
    }
    return data;
  }

  function validateTraces(value) {
    const data = requireRecord(value, 'trace history');
    if (!Array.isArray(data.records) || !Number.isInteger(data.count) || data.count < 0) {
      throw new SatQueryApiError('The API returned an invalid trace-history response.', { kind: 'invalid_response' });
    }
    return data;
  }

  function validateVerification(value) {
    const data = requireRecord(value, 'trace verification');
    if (typeof data.verified !== 'boolean') {
      throw new SatQueryApiError('The API returned an invalid trace-verification response.', { kind: 'invalid_response' });
    }
    requireString(data, 'message', 'trace verification');
    return { verified: data.verified, message: data.message };
  }

  function resourceUrl(pathSegment, suffix) {
    if (typeof pathSegment !== 'string' || !pathSegment.trim()) {
      throw new SatQueryApiError('A non-empty resource identifier is required.', { kind: 'validation' });
    }
    return `${apiBase()}${suffix(encodeURIComponent(pathSegment.trim()))}`;
  }

  async function getHealth() {
    return validateHealth(await request('/api/health'));
  }

  async function analyzeScene(payload) {
    const data = requireRecord(payload, 'analysis request');
    requireString(data, 'scene_id', 'analysis request');
    requireString(data, 'question', 'analysis request');
    if (data.sensor !== undefined && typeof data.sensor !== 'string') {
      throw new SatQueryApiError('The analysis sensor value must be text.', { kind: 'validation' });
    }
    return validateAnalysis(await request('/api/analyze', { method: 'POST', body: data }));
  }

  function getSceneImageUrl(sceneId) {
    return resourceUrl(sceneId, id => `/api/scenes/${id}/image`);
  }

  async function getResolution() {
    return validateResolution(await request('/api/resolution'));
  }

  async function getSar(scene) {
    const url = resourceUrl(scene, id => `/api/sar/${id}`);
    return validateSar(await request(url.slice(apiBase().length)));
  }

  function getSarImageUrl(scene) {
    return resourceUrl(scene, id => `/api/sar/${id}/image`);
  }

  async function getTraces() {
    return validateTraces(await request('/api/traces'));
  }

  async function verifyTraces(records) {
    if (records !== undefined && !Array.isArray(records)) {
      throw new SatQueryApiError('Trace records must be supplied as a list.', { kind: 'validation' });
    }
    // The current endpoint verifies its process-local chain and accepts no body.
    return validateVerification(await request('/api/traces/verify', { method: 'POST' }));
  }

  global.SatQueryApi = Object.freeze({
    get baseUrl() { return apiBase(); },
    getHealth,
    analyzeScene,
    getSceneImageUrl,
    getResolution,
    getSar,
    getSarImageUrl,
    getTraces,
    verifyTraces,
    SatQueryApiError
  });
})(window);

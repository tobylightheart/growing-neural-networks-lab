function fmt(value) {
  return Number(value).toFixed(3);
}

function truthyPredictions(predictions) {
  return JSON.stringify(predictions) === JSON.stringify([0, 1, 1, 0]);
}

function rowInputs(row) {
  return row.x.map(Number);
}

function renderTrace(result) {
  const baseline = result.baseline;
  const candidate = result.candidate;
  const grown = result.grown_model;
  const comparison = result.comparison;
  const trace = result.growth_trace;

  document.getElementById('baseline').textContent = `weights: [${baseline.weights.map(fmt).join(', ')}]\noutputs: ${baseline.outputs.map(fmt).join(', ')}\nmse: ${fmt(baseline.mse)}`;
  document.getElementById('candidate').textContent = `bias: ${fmt(candidate.bias)}\nweights: [${candidate.weights.map(fmt).join(', ')}]\n|corr(hidden, residual)| ≈ ${fmt(candidate.residual_correlation)}`;
  document.getElementById('grown').textContent = `weights: [${grown.weights.map(fmt).join(', ')}]\npredictions: ${grown.predictions.join(', ')}\nsolves XOR: ${truthyPredictions(grown.predictions)}`;
  document.getElementById('comparison').textContent = `MSE reduction: ${fmt(comparison.mse_reduction)}\nerror reduction factor: ${comparison.error_reduction_factor.toLocaleString(undefined, { maximumFractionDigits: 0 })}×\nminimum threshold margin: ${fmt(comparison.min_output_margin)}\nhidden feature frozen before refit: ${comparison.hidden_feature_frozen_before_refit}`;

  document.getElementById('trace').innerHTML = trace.map(row => {
    const [x1, x2] = rowInputs(row);
    return `
  <tr>
    <td>${x1}</td>
    <td>${x2}</td>
    <td>${fmt(row.target)}</td>
    <td>${fmt(row.baseline_output)}</td>
    <td>${fmt(row.residual)}</td>
    <td>${fmt(row.hidden_value)}</td>
    <td>${fmt(row.grown_output)}</td>
    <td>${fmt(row.margin_from_threshold)}</td>
    <td>${row.prediction}</td>
  </tr>`;
  }).join('');
}

function renderError(error) {
  const message = `Could not load trace.json: ${error.message}`;
  document.getElementById('baseline').textContent = message;
  document.getElementById('candidate').textContent = 'Trace unavailable';
  document.getElementById('grown').textContent = 'Trace unavailable';
  document.getElementById('comparison').textContent = 'Trace unavailable';
  document.getElementById('trace').innerHTML = `<tr><td colspan="9">${message}</td></tr>`;
}

fetch('trace.json')
  .then(response => {
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  })
  .then(renderTrace)
  .catch(renderError);

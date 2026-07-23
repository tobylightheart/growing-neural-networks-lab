function fmt(value) {
  return Number(value).toFixed(1);
}

function vector(values) {
  return `[${values.map(fmt).join(', ')}]`;
}

function renderTrace(result) {
  const policy = result.policy;
  const final = result.final_model;
  document.getElementById('policy').textContent = `weights: ${policy.weight_order.join(', ')}
initial: ${vector(policy.initial_weights)}
learning rate: ${policy.learning_rate}
positive when: ${policy.positive_when}
maximum epochs: ${policy.max_epochs}`;
  document.getElementById('final').textContent = `weights: ${vector(final.weights)}
predictions: ${final.predictions.join(', ')}
targets: ${final.targets.join(', ')}
epochs run: ${final.epochs_run}
converged: ${final.converged}`;

  document.getElementById('epochs').innerHTML = result.epoch_trace.map(row => `
    <tr>
      <td>${row.epoch}</td>
      <td>${row.errors}</td>
      <td>${vector(row.weights_after)}</td>
      <td>${row.predictions_after.join(', ')}</td>
      <td>${row.converged}</td>
    </tr>`).join('');

  document.getElementById('samples').innerHTML = result.sample_trace.map(row => `
    <tr>
      <td>${row.epoch}</td>
      <td>${row.x.join(', ')}</td>
      <td>${row.target}</td>
      <td>${fmt(row.activation)}</td>
      <td>${row.prediction}</td>
      <td>${row.error}</td>
      <td>${vector(row.update)}</td>
      <td>${vector(row.weights_after)}</td>
    </tr>`).join('');
}

function renderError(error) {
  const message = `Could not load trace.json: ${error.message}`;
  document.getElementById('policy').textContent = message;
  document.getElementById('final').textContent = 'Trace unavailable';
  document.getElementById('epochs').innerHTML = `<tr><td colspan="5">${message}</td></tr>`;
  document.getElementById('samples').innerHTML = `<tr><td colspan="8">${message}</td></tr>`;
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

const xor = [
  { x: [0, 0], target: 0 },
  { x: [0, 1], target: 1 },
  { x: [1, 0], target: 1 },
  { x: [1, 1], target: 0 }
];

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Values mirrored from the deterministic Python run. Keeping them literal makes
// the page static and inspectable while the Python script remains the canonical
// runnable experiment.
const result = {
  baseline: {
    outputs: [0.49999962500053124, 0.4999998750000938, 0.4999998750000938, 0.5000001249996563],
    mse: 0.25000000000004685
  },
  candidate: {
    bias: -2,
    weights: [-12, -12],
    residualCorrelation: 0.577,
    hiddenValues: xor.map(row => sigmoid(-2 - 12 * row.x[0] - 12 * row.x[1]))
  },
  grown: {
    weights: [2.000, -1.000, -1.000, -16.774],
    mse: 0.000000020512956764548918
  }
};

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

const grownOutputs = xor.map((row, i) => dot(result.grown.weights, [1, row.x[0], row.x[1], result.candidate.hiddenValues[i]]));
const predictions = grownOutputs.map(value => value >= 0.5 ? 1 : 0);
const outputMargins = grownOutputs.map((value, i) => xor[i].target === 1 ? value - 0.5 : 0.5 - value);

function fmt(value) {
  return Number(value).toFixed(3);
}

document.getElementById('baseline').textContent = `outputs: ${result.baseline.outputs.map(fmt).join(', ')}\nmse: ${fmt(result.baseline.mse)}`;
document.getElementById('candidate').textContent = `bias: ${result.candidate.bias}\nweights: [${result.candidate.weights.join(', ')}]\n|corr(hidden, residual)| ≈ ${fmt(result.candidate.residualCorrelation)}`;
document.getElementById('grown').textContent = `predictions: ${predictions.join(', ')}\nsolves XOR: ${JSON.stringify(predictions) === JSON.stringify([0,1,1,0])}`;
document.getElementById('comparison').textContent = `MSE reduction: ${fmt(result.baseline.mse - result.grown.mse)}\nerror reduction factor: ${(result.baseline.mse / result.grown.mse).toLocaleString(undefined, { maximumFractionDigits: 0 })}×\nminimum threshold margin: ${fmt(Math.min(...outputMargins))}\nhidden feature frozen before refit: true`;

document.getElementById('trace').innerHTML = xor.map((row, i) => `
  <tr>
    <td>${row.x[0]}</td>
    <td>${row.x[1]}</td>
    <td>${row.target}</td>
    <td>${fmt(result.baseline.outputs[i])}</td>
    <td>${fmt(row.target - result.baseline.outputs[i])}</td>
    <td>${fmt(result.candidate.hiddenValues[i])}</td>
    <td>${fmt(grownOutputs[i])}</td>
    <td>${fmt(outputMargins[i])}</td>
    <td>${predictions[i]}</td>
  </tr>`).join('');

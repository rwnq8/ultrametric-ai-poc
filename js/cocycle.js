/**
 * Cocycle Cognitive Architecture — ultrametric verification.
 * Prime basis, p-adic valuations, ultrametric distance,
 * strong triangle inequality, pairwise attention.
 */

const PRIME_BASIS = { good: 2, bad: 3, not: 5, very: 7, but: 11 };
const PRIME_NAMES = Object.keys(PRIME_BASIS);

function valuation(n, p) {
  if (n <= 0) return 0;
  let v = 0;
  while (n % p === 0) { n = Math.floor(n / p); v++; }
  return v;
}

function toValuationVector(n) {
  return PRIME_NAMES.map(pn => valuation(n, PRIME_BASIS[pn]));
}

function ultrametricDistance(v1, v2) {
  let max = 0;
  for (let i = 0; i < v1.length; i++) max = Math.max(max, Math.abs(v1[i] - v2[i]));
  return max;
}

function checkStrongTriangle(v1, v2, v3) {
  const d12 = ultrametricDistance(v1, v2), d13 = ultrametricDistance(v1, v3), d23 = ultrametricDistance(v2, v3);
  const ok = d12 <= Math.max(d13, d23);
  return { ok, d12, d13, d23, max: Math.max(d13, d23), isIsosceles: (d12 === d13 || d12 === d23 || d13 === d23) };
}

function pairwiseAttentionMatrix(valuations, temperature = 1.0) {
  const n = valuations.length;
  const m = valuations[0].length;
  const dist = Array(n).fill(null).map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      dist[i][j] = ultrametricDistance(valuations[i], valuations[j]);
  
  const sim = dist.map(row => row.map(d => Math.exp(-d / Math.max(temperature, 0.01))));
  const attn = sim.map((row, i) => {
    const sum = row.reduce((a, b) => a + b, 0) || 1;
    return row.map(v => v / sum);
  });
  return { attn, dist };
}

function globalCocycleConsistency(valuations, tokens) {
  let total = 0, sat = 0, violations = [];
  const n = valuations.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        total++;
        const r = checkStrongTriangle(valuations[i], valuations[j], valuations[k]);
        if (r.ok) sat++; else violations.push(`${tokens[i]}-${tokens[j]}-${tokens[k]}`);
      }
    }
  }
  return { total, sat, ratio: total > 0 ? sat / total : 1, violations };
}

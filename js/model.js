/**
 * Model — Word-to-prime mapping and attention models.
 * Hardcoded lookup table for demo purposes (no WordNet in browser).
 */

// Pre-built word-to-prime mapping for demonstration
const WORD_PRIME_MAP = {
  // Positive
  good: ['good'], great: ['good'], excellent: ['good'], wonderful: ['good'],
  fantastic: ['good'], amazing: ['good'], perfect: ['good'], lovely: ['good'],
  beautiful: ['good'], happy: ['good'], positive: ['good'], nice: ['good'],
  awesome: ['good'], brilliant: ['good'],
  // Negative
  bad: ['bad'], terrible: ['bad'], awful: ['bad'], horrible: ['bad'],
  worst: ['bad'], poor: ['bad'], negative: ['bad'], ugly: ['bad'],
  boring: ['bad'], stupid: ['bad'], sad: ['bad'], disappointing: ['bad'],
  // Negation
  not: ['not'], "n't": ['not'], never: ['not'], no: ['not'], none: ['not'],
  // Intensifiers
  very: ['very'], extremely: ['very'], really: ['very'], highly: ['very'],
  quite: ['very'], so: ['very'], too: ['very'],
  // Contrast
  but: ['but'], however: ['but'], although: ['but'], yet: ['but'], though: ['but'],
  // Context-dependent (assigned by heuristic)
  movie: ['good'], film: ['good'], acting: ['good'], script: ['good'],
  story: ['good'], performance: ['good'], plot: ['good'],
  waste: ['bad', 'not'], disaster: ['bad', 'very'],
};

function wordToPrimes(word) {
  const w = word.toLowerCase();
  // Check direct match
  if (WORD_PRIME_MAP[w]) return WORD_PRIME_MAP[w];
  
  // Heuristic: words containing known roots
  if (w.includes('good') || w.includes('great') || w.includes('excel')) return ['good'];
  if (w.includes('bad') || w.includes('terrib') || w.includes('awful')) return ['bad'];
  if (w.includes('love') || w.includes('wonder') || w.includes('amaz')) return ['good'];
  if (w.includes('hate') || w.includes('worst') || w.includes('horrib')) return ['bad'];
  
  return []; // Unclassified
}

function wordToProduct(word) {
  const primes = wordToPrimes(word);
  let prod = 1;
  for (const p of primes) if (p in PRIME_BASIS) prod *= PRIME_BASIS[p];
  return prod;
}

function sentenceToValuations(sentence) {
  const tokens = sentence.toLowerCase().match(/[\w']+|[^\w\s]/g) || [];
  const vals = [], prods = [], rawTokens = [];
  for (const t of tokens) {
    const prod = wordToProduct(t);
    prods.push(prod);
    vals.push(toValuationVector(prod));
    rawTokens.push(t);
  }
  return { tokens: rawTokens, valuations: vals, products: prods };
}

function tokenToExpr(token) {
  const primes = wordToPrimes(token);
  if (primes.length === 0) return mark();
  let e = mark();
  for (let i = 0; i < primes.length; i++) e = enclose(e);
  return e;
}

function sentenceToExprs(sentence) {
  const tokens = (sentence.toLowerCase().match(/[\w']+|[^\w\s]/g) || []);
  return tokens.map(t => tokenToExpr(t));
}

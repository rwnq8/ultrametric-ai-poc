/**
 * Distinction Calculus — Spencer-Brown primitives in JavaScript.
 * Marks (#), Enclosures ([ ]), Calling (##→#), Crossing ([[A]]→A).
 * No floating point. Pure syntactic pattern operations.
 */

class Expr {
  constructor(raw) {
    this.raw = raw;
    this._validate();
  }

  _validate() {
    let depth = 0;
    for (const c of this.raw) {
      if (c === '[') depth++;
      else if (c === ']') depth--;
      if (depth < 0) throw new Error(`Unbalanced: ${this.raw}`);
    }
    if (depth !== 0) throw new Error(`Unclosed: ${this.raw}`);
  }

  get depth() {
    let d = 0, max = 0;
    for (const c of this.raw) { if (c === '[') { d++; max = Math.max(max, d); } else if (c === ']') d--; }
    return max;
  }

  get markCount() { return (this.raw.match(/#/g) || []).length; }

  reduceCalling() {
    let r = this.raw;
    while (r.includes('##')) r = r.replace(/##/g, '#');
    return new Expr(r);
  }

  reduceCrossing() {
    let r = this.raw;
    let prev = '';
    while (prev !== r) {
      prev = r;
      r = r.replace(/\[\[([^\[\]]*(?:\[[^\]]*\][^\[\]]*)*)\]\]/g, '$1');
    }
    return new Expr(r);
  }

  normalForm() {
    let cur = this.reduceCalling().reduceCrossing(), prev = '';
    while (prev !== cur.raw) { prev = cur.raw; cur = cur.reduceCalling().reduceCrossing(); }
    return cur;
  }

  isStable() { return this.normalForm().raw === this.raw; }

  toTreePath() {
    const path = [];
    let depth = 0;
    for (let i = 0; i < this.raw.length; i++) {
      if (this.raw[i] === '[') depth++;
      else if (this.raw[i] === ']') depth--;
      else if (this.raw[i] === '#') path.push(depth);
    }
    return path;
  }
}

function mark() { return new Expr('#'); }
function enclose(e) { return new Expr(`[${e.raw}]`); }
function juxtapose(...exprs) { return new Expr(exprs.map(e => e.raw).join('')); }

function treeDistance(e1, e2) {
  const p1 = e1.toTreePath(), p2 = e2.toTreePath();
  const ml = Math.max(p1.length, p2.length);
  const a = [...p1, ...Array(ml - p1.length).fill(0)];
  const b = [...p2, ...Array(ml - p2.length).fill(0)];
  let lca = 0;
  for (let i = 0; i < ml; i++) if (a[i] === b[i] && a[i] > 0) lca = Math.max(lca, a[i]);
  const d1 = a.filter(d => d > lca).length, d2 = b.filter(d => d > lca).length;
  return d1 + d2;
}

function strongTriangleInequality(e1, e2, e3) {
  const d12 = treeDistance(e1, e2), d13 = treeDistance(e1, e3), d23 = treeDistance(e2, e3);
  return d12 <= Math.max(d13, d23);
}

// Known stable particle patterns
const STABLE_PATTERNS = {
  photon: new Expr('[#]'),
  electron: new Expr('[# [#]]'),
  up_quark: new Expr('[[#] #]'),
  down_quark: new Expr('[[#] [#] #]'),
  w_boson: new Expr('[[#] [#]]'),
  z_higgs: new Expr('[[#] [#] [#]]'),
};

// Semantic primes encoded as patterns
const SEMANTIC_PRIME_PATTERNS = {
  good: new Expr('[#]'),
  bad: new Expr('[# [#]]'),
  not: new Expr('[[#]]'),
  very: new Expr('[##]'),
  but: new Expr('[# [#] [#]]'),
};

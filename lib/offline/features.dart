// The 406 photo features, the same as ml/src/features.py (color, GLCM, LBP, HOG), so the
// trained models can run on the phone. Checked against Python in test/offline_parity_test.dart.
import 'dart:math' as math;
import 'dart:typed_data';

import 'image_ops.dart';

const featureLength = 406;
const _imgSize = 256, _hogSize = 128, _glcmLevels = 32;

/// Features for one RGB photo of any size.
Float32List extractFeatures(Img rgb) {
  final small = resizeArea(rgb, _imgSize, _imgSize);
  final gray = toGray(small);
  final out = Float32List(featureLength);
  var i = 0;
  for (final list in [_color(small), _glcm(gray), _lbp(gray), _hog(gray)]) {
    for (final v in list) {
      out[i++] = v;
    }
  }
  assert(i == featureLength);
  return out;
}

// --- color: mean and std of R, G, B, H, S, V, then H/S/V histograms -------------------------

List<double> _color(Img rgb) {
  final hsv = toHsv(rgb);
  final n = rgb.width * rgb.height;
  final stats = <double>[];
  for (final img in [rgb, hsv]) {
    final means = <double>[];
    for (var c = 0; c < 3; c++) {
      var s = 0.0;
      for (var i = 0; i < n; i++) {
        s += img.data[i * 3 + c];
      }
      means.add(s / n);
    }
    stats.addAll(means.map((m) => m / 255.0));
    for (var c = 0; c < 3; c++) {
      var s = 0.0;
      for (var i = 0; i < n; i++) {
        final d = img.data[i * 3 + c] - means[c];
        s += d * d;
      }
      stats.add(math.sqrt(s / n) / 255.0);
    }
  }
  final hist = <double>[];
  for (final (ch, bins, upper) in const [(0, 16, 180), (1, 8, 256), (2, 8, 256)]) {
    final counts = List<int>.filled(bins, 0);
    for (var i = 0; i < n; i++) {
      counts[math.min(bins - 1, hsv.data[i * 3 + ch] * bins ~/ upper)]++;
    }
    hist.addAll(counts.map((c) => c / n));
  }
  return [...stats, ...hist];
}

// --- GLCM: 32 gray levels, distances 1 and 3, angles 0/45/90/135, averaged over angles -------

const _glcmOffsets = {
  1: [(0, 1), (1, 1), (1, 0), (1, -1)],
  3: [(0, 3), (2, 2), (3, 0), (2, -2)],
};

List<double> _glcm(Img gray) {
  final rows = gray.height, cols = gray.width;
  final q = Uint8List(rows * cols);
  for (var i = 0; i < q.length; i++) {
    q[i] = gray.data[i] ~/ (256 ~/ _glcmLevels);
  }
  const L = _glcmLevels;
  // props[distance] = 5 properties, each summed over the 4 angles
  final props = {for (final d in _glcmOffsets.keys) d: List<double>.filled(5, 0)};
  for (final d in _glcmOffsets.keys) {
    for (final (orow, ocol) in _glcmOffsets[d]!) {
      final counts = Float64List(L * L);
      for (var r = math.max(0, -orow); r < math.min(rows, rows - orow); r++) {
        for (var c = math.max(0, -ocol); c < math.min(cols, cols - ocol); c++) {
          counts[q[r * cols + c] * L + q[(r + orow) * cols + c + ocol]] += 1;
        }
      }
      final p = Float64List(L * L);
      var total = 0.0;
      for (var i = 0; i < L; i++) {
        for (var j = 0; j < L; j++) {
          p[i * L + j] = counts[i * L + j] + counts[j * L + i]; // symmetric
          total += p[i * L + j];
        }
      }
      for (var k = 0; k < p.length; k++) {
        p[k] /= total;
      }
      var contrast = 0.0, dissim = 0.0, homog = 0.0, asm = 0.0, meanI = 0.0, meanJ = 0.0;
      for (var i = 0; i < L; i++) {
        for (var j = 0; j < L; j++) {
          final v = p[i * L + j], diff = (i - j).toDouble();
          contrast += v * diff * diff;
          dissim += v * diff.abs();
          homog += v / (1.0 + diff * diff);
          asm += v * v;
          meanI += i * v;
          meanJ += j * v;
        }
      }
      var varI = 0.0, varJ = 0.0, cov = 0.0;
      for (var i = 0; i < L; i++) {
        for (var j = 0; j < L; j++) {
          final v = p[i * L + j], di = i - meanI, dj = j - meanJ;
          varI += v * di * di;
          varJ += v * dj * dj;
          cov += v * di * dj;
        }
      }
      final si = math.sqrt(varI), sj = math.sqrt(varJ);
      final corr = (si < 1e-15 || sj < 1e-15) ? 1.0 : cov / (si * sj);
      final values = [contrast, dissim, homog, math.sqrt(asm), corr];
      for (var k = 0; k < 5; k++) {
        props[d]![k] += values[k];
      }
    }
  }
  return [
    for (var k = 0; k < 5; k++)
      for (final d in const [1, 3]) props[d]![k] / 4,
  ];
}

// --- LBP: "uniform" local binary patterns (8 points radius 1, 16 points radius 2) -----------

// Sample positions from scikit-image (rounded to 5 decimals there), copied from Python.
const _lbp8r = [-0.0, -0.70711, -1.0, -0.70711, -0.0, 0.70711, 1.0, 0.70711];
const _lbp8c = [1.0, 0.70711, 0.0, -0.70711, -1.0, -0.70711, -0.0, 0.70711];
const _lbp16r = [-0.0, -0.76537, -1.41421, -1.84776, -2.0, -1.84776, -1.41421, -0.76537,
  -0.0, 0.76537, 1.41421, 1.84776, 2.0, 1.84776, 1.41421, 0.76537];
const _lbp16c = [2.0, 1.84776, 1.41421, 0.76537, 0.0, -0.76537, -1.41421, -1.84776,
  -2.0, -1.84776, -1.41421, -0.76537, -0.0, 0.76537, 1.41421, 1.84776];

List<double> _lbp(Img gray) {
  final rows = gray.height, cols = gray.width, img = gray.data;
  double px(int r, int c) => (r >= 0 && r < rows && c >= 0 && c < cols) ? img[r * cols + c].toDouble() : 0.0;
  final out = <double>[];
  for (final (rp, cp) in const [(_lbp8r, _lbp8c), (_lbp16r, _lbp16c)]) {
    final P = rp.length;
    final counts = List<int>.filled(P + 2, 0);
    final bits = List<int>.filled(P, 0);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final center = img[r * cols + c].toDouble();
        var ones = 0;
        for (var p = 0; p < P; p++) {
          final y = r + rp[p], x = c + cp[p];
          final minr = y.floor(), minc = x.floor(), maxr = y.ceil(), maxc = x.ceil();
          final dr = y - minr, dc = x - minc;
          final top = (1 - dc) * px(minr, minc) + dc * px(minr, maxc);
          final bottom = (1 - dc) * px(maxr, minc) + dc * px(maxr, maxc);
          final v = (1 - dr) * top + dr * bottom;
          bits[p] = v - center >= 0 ? 1 : 0;
          ones += bits[p];
        }
        var changes = 0;
        for (var p = 0; p < P - 1; p++) {
          if (bits[p] != bits[p + 1]) changes++;
        }
        counts[changes <= 2 ? ones : P + 1]++;
      }
    }
    final n = rows * cols;
    out.addAll(counts.map((k) => k / n));
  }
  return out;
}

// --- HOG: 128x128 gray, 9 orientations, 32 px cells, 2x2 blocks, L2-Hys ---------------------

List<double> _hog(Img gray256) {
  final img = resizeArea(gray256, _hogSize, _hogSize);
  const n = _hogSize, cell = 32, cells = n ~/ cell;
  final mag = Float64List(n * n), ori = Float64List(n * n);
  for (var r = 0; r < n; r++) {
    for (var c = 0; c < n; c++) {
      final gr = (r == 0 || r == n - 1) ? 0.0 : (img.data[(r + 1) * n + c] - img.data[(r - 1) * n + c]).toDouble();
      final gc = (c == 0 || c == n - 1) ? 0.0 : (img.data[r * n + c + 1] - img.data[r * n + c - 1]).toDouble();
      mag[r * n + c] = math.sqrt(gc * gc + gr * gr);
      ori[r * n + c] = (math.atan2(gr, gc) * (180.0 / math.pi)) % 180;
    }
  }
  final hist = Float64List(cells * cells * 9);
  for (var i = 0; i < 9; i++) {
    final start = 20.0 * (i + 1), end = 20.0 * i;
    for (var cr = 0; cr < cells; cr++) {
      for (var cc = 0; cc < cells; cc++) {
        var total = 0.0;
        for (var r = cr * cell; r < cr * cell + cell; r++) {
          for (var c = cc * cell; c < cc * cell + cell; c++) {
            final o = ori[r * n + c];
            if (o >= start || o < end) continue;
            total += mag[r * n + c];
          }
        }
        hist[(cr * cells + cc) * 9 + i] = total / (cell * cell);
      }
    }
  }
  const eps = 1e-5;
  final out = <double>[];
  for (var br = 0; br < cells - 1; br++) {
    for (var bc = 0; bc < cells - 1; bc++) {
      final block = <double>[
        for (var r = br; r < br + 2; r++)
          for (var c = bc; c < bc + 2; c++)
            for (var i = 0; i < 9; i++) hist[(r * cells + c) * 9 + i],
      ];
      var s = 0.0;
      for (final v in block) {
        s += v * v;
      }
      var b = [for (final v in block) math.min(v / math.sqrt(s + eps * eps), 0.2)];
      s = 0.0;
      for (final v in b) {
        s += v * v;
      }
      b = [for (final v in b) v / math.sqrt(s + eps * eps)];
      out.addAll(b);
    }
  }
  return out;
}

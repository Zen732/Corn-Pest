// Damage severity on the phone, following ml/src/severity.py step by step: background removal,
// holes, K-Means color clusters in Lab, the green-is-healthy rule, the edge band, damage %.
// Every step except K-Means gives exactly Python's pixels. scikit-learn's K-Means picks its
// starting points with Python's random generator, which can't be reproduced here, so this uses
// its own fixed-seed K-Means; test/offline_parity_test.dart measures how often the grade agrees.
import 'dart:math' as math;
import 'dart:typed_data';

import 'image_ops.dart';

const workSize = 512, kClusters = 4, fitPixels = 20000, nInit = 4;
const healthyBelow = 5.0, lowUpTo = 20.0, moderateUpTo = 40.0;
const minPlantShare = 0.05, edgeBand = 2, hueMin = 105.0, chromaMin = 6.0;

class Severity {
  const Severity({this.damagePercent, required this.level, required this.plantShare, this.greenShare = 0, this.holePercent = 0});
  final double? damagePercent; // null when no plant was found
  final String level; // Healthy, Low, Moderate, High, Unclear
  final double plantShare, greenShare, holePercent;
}

String grade(double pct) =>
    pct < healthyBelow ? 'Healthy' : (pct <= lowUpTo ? 'Low' : (pct <= moderateUpTo ? 'Moderate' : 'High'));

/// Like Python's round(x, digits): based on the exact stored value (0.8995 is stored a hair
/// under, so it becomes 0.899, not 0.900).
double roundLikePython(double x, int digits) => double.parse(x.toStringAsFixed(digits));

double _round(double x, int digits) => roundLikePython(x, digits);

Severity assessSeverity(Img input) {
  var rgb = input;
  final scale = workSize / math.max(rgb.width, rgb.height);
  if (scale < 1) rgb = resizeAreaBy(rgb, scale);
  rgb = gaussianBlur3(rgb);
  final w = rgb.width, h = rgb.height, total = w * h;

  final (background, holes) = _backgroundMask(rgb);
  final plant = Uint8List(total);
  var plantCount = 0;
  for (var i = 0; i < total; i++) {
    if (background[i] == 0 && holes[i] == 0) {
      plant[i] = 1;
      plantCount++;
    }
  }
  if (plantCount < minPlantShare * total) {
    return Severity(level: 'Unclear', plantShare: plantCount / total);
  }

  // Lab as float, like severity.py's _lab: L 0-100, a and b centred on 0, weighted (0.5, 1, 1).
  final lab = toLab(rgb);
  final f = Float32List(1);
  double f32(double v) {
    f[0] = v;
    return f[0];
  }

  final lScale = f32(100.0 / 255.0);
  final idx = <int>[for (var i = 0; i < total; i++) if (plant[i] == 1) i];
  final px = Float32List(idx.length * 3);
  for (var n = 0; n < idx.length; n++) {
    final i = idx[n];
    px[n * 3] = f32(f32(lab.data[i * 3] * lScale) * 0.5);
    px[n * 3 + 1] = (lab.data[i * 3 + 1] - 128).toDouble();
    px[n * 3 + 2] = (lab.data[i * 3 + 2] - 128).toDouble();
  }
  final centers = _kmeans(px);
  final healthyCluster = [
    for (var k = 0; k < kClusters; k++) _isHealthy(centers[k * 3] / 0.5, centers[k * 3 + 1], centers[k * 3 + 2]),
  ];

  final damaged = Uint8List(total);
  for (var n = 0; n < idx.length; n++) {
    if (!healthyCluster[_nearest(px, n, centers)]) damaged[idx[n]] = 1;
  }

  // Leave a thin band along the plant's outline out of the count (blended colors).
  final region = Uint8List(total);
  for (var i = 0; i < total; i++) {
    region[i] = (plant[i] == 1 || holes[i] == 1) ? 1 : 0;
  }
  final inner = _erode(region, w, h, edgeBand);
  var plantN = 0, damagedN = 0, holeN = 0, greenN = 0;
  for (var i = 0; i < total; i++) {
    final band = region[i] == 1 && inner[i] == 0;
    final p = plant[i] == 1 && !band, d = damaged[i] == 1 && !band;
    if (p) plantN++;
    if (d) damagedN++;
    if (holes[i] == 1) holeN++;
    if (p && !d) greenN++;
  }
  final plantArea = plantN + holeN;
  final pct = 100.0 * (damagedN + holeN) / plantArea;
  return Severity(
    damagePercent: _round(pct, 1),
    level: grade(pct),
    plantShare: _round(plantArea / total, 3),
    greenShare: _round(greenN / total, 3),
    holePercent: _round(100.0 * holeN / plantArea, 1),
  );
}

bool _isHealthy(double l, double a, double b) {
  final chroma = math.sqrt(a * a + b * b);
  final hue = (math.atan2(b, a) * 180 / math.pi) % 360;
  return chroma >= chromaMin && hue >= hueMin && hue <= 220 && l >= 20;
}

/// White/grey tables, black backgrounds and sky touching the photo's edge are background;
/// the same colors enclosed by the plant are holes or dead spots (counted as damage).
(Uint8List, Uint8List) _backgroundMask(Img rgb) {
  final w = rgb.width, h = rgb.height, total = w * h;
  final hsv = toHsv(rgb);
  final cand = Uint8List(total);
  for (var i = 0; i < total; i++) {
    final hue = hsv.data[i * 3], s = hsv.data[i * 3 + 1] / 255.0, v = hsv.data[i * 3 + 2] / 255.0;
    final white = s < 0.15 && v > 0.80, black = v < 0.12, grey = s < 0.10 && v >= 0.12;
    final sky = hue >= 90 && hue <= 130 && s > 0.20 && v > 0.35;
    if (white || black || grey || sky) cand[i] = 1;
  }
  // Candidate regions (4-connected) that touch the edge.
  final edge = Uint8List(total);
  final stack = <int>[];
  void seed(int i) {
    if (cand[i] == 1 && edge[i] == 0) {
      edge[i] = 1;
      stack.add(i);
    }
  }

  for (var x = 0; x < w; x++) {
    seed(x);
    seed((h - 1) * w + x);
  }
  for (var y = 0; y < h; y++) {
    seed(y * w);
    seed(y * w + w - 1);
  }
  while (stack.isNotEmpty) {
    final i = stack.removeLast(), x = i % w, y = i ~/ w;
    if (x > 0) seed(i - 1);
    if (x < w - 1) seed(i + 1);
    if (y > 0) seed(i - w);
    if (y < h - 1) seed(i + w);
  }
  // scipy binary_opening(iterations=2): two erosions, then two dilations (cross shape).
  final background = _dilate(_erode(edge, w, h, 2), w, h, 2);
  final holes = Uint8List(total);
  for (var i = 0; i < total; i++) {
    holes[i] = (cand[i] == 1 && background[i] == 0) ? 1 : 0;
  }
  return (background, holes);
}

/// scipy.ndimage.binary_erosion with the cross shape; outside the photo counts as 0.
Uint8List _erode(Uint8List m, int w, int h, int times) {
  var cur = m;
  for (var t = 0; t < times; t++) {
    final next = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final i = y * w + x;
        next[i] = (cur[i] == 1 && x > 0 && cur[i - 1] == 1 && x < w - 1 && cur[i + 1] == 1 &&
                y > 0 && cur[i - w] == 1 && y < h - 1 && cur[i + w] == 1)
            ? 1
            : 0;
      }
    }
    cur = next;
  }
  return cur;
}

/// scipy.ndimage.binary_dilation with the cross shape.
Uint8List _dilate(Uint8List m, int w, int h, int times) {
  var cur = m;
  for (var t = 0; t < times; t++) {
    final next = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final i = y * w + x;
        next[i] = (cur[i] == 1 ||
                (x > 0 && cur[i - 1] == 1) ||
                (x < w - 1 && cur[i + 1] == 1) ||
                (y > 0 && cur[i - w] == 1) ||
                (y < h - 1 && cur[i + w] == 1))
            ? 1
            : 0;
      }
    }
    cur = next;
  }
  return cur;
}

// --- K-Means: fixed-seed k-means++ starts, 4 runs, Lloyd iterations; best (lowest error) kept --

class _Rng {
  _Rng(int seed) : _s = seed & 0x7fffffff;
  int _s;
  double next() {
    _s = (_s * 1103515245 + 12345) & 0x7fffffff;
    return _s / 0x80000000;
  }

  int nextInt(int n) => (next() * n).floor() % n;
}

double _d2(Float32List a, int i, Float64List c, int k) {
  final x = a[i * 3] - c[k * 3], y = a[i * 3 + 1] - c[k * 3 + 1], z = a[i * 3 + 2] - c[k * 3 + 2];
  return x * x + y * y + z * z;
}

int _nearest(Float32List px, int n, Float64List centers) {
  var best = 0, bestD = double.infinity;
  for (var k = 0; k < kClusters; k++) {
    final d = _d2(px, n, centers, k);
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

Float64List _kmeans(Float32List all) {
  final nAll = all.length ~/ 3;
  final rng = _Rng(0);
  // Fit on at most fitPixels pixels, picked with a fixed seed (like severity.py's sample).
  Float32List data;
  if (nAll > fitPixels) {
    final order = List<int>.generate(nAll, (i) => i);
    for (var i = 0; i < fitPixels; i++) {
      final j = i + rng.nextInt(nAll - i);
      final t = order[i];
      order[i] = order[j];
      order[j] = t;
    }
    data = Float32List(fitPixels * 3);
    for (var i = 0; i < fitPixels; i++) {
      for (var c = 0; c < 3; c++) {
        data[i * 3 + c] = all[order[i] * 3 + c];
      }
    }
  } else {
    data = all;
  }
  final n = data.length ~/ 3;
  // scikit-learn's tolerance: 1e-4 x the mean variance of the features.
  var tol = 0.0;
  for (var c = 0; c < 3; c++) {
    var m = 0.0;
    for (var i = 0; i < n; i++) {
      m += data[i * 3 + c];
    }
    m /= n;
    var v = 0.0;
    for (var i = 0; i < n; i++) {
      final d = data[i * 3 + c] - m;
      v += d * d;
    }
    tol += v / n;
  }
  tol = tol / 3 * 1e-4;

  Float64List? best;
  var bestInertia = double.infinity;
  for (var run = 0; run < nInit; run++) {
    final centers = _plusPlus(data, n, rng);
    final labels = Int32List(n);
    for (var iter = 0; iter < 300; iter++) {
      for (var i = 0; i < n; i++) {
        labels[i] = _nearest(data, i, centers);
      }
      final sums = Float64List(kClusters * 3), counts = Int32List(kClusters);
      for (var i = 0; i < n; i++) {
        final k = labels[i];
        counts[k]++;
        for (var c = 0; c < 3; c++) {
          sums[k * 3 + c] += data[i * 3 + c];
        }
      }
      var shift = 0.0;
      for (var k = 0; k < kClusters; k++) {
        if (counts[k] == 0) continue; // keep an empty cluster where it is
        for (var c = 0; c < 3; c++) {
          final nc = sums[k * 3 + c] / counts[k];
          shift += (nc - centers[k * 3 + c]) * (nc - centers[k * 3 + c]);
          centers[k * 3 + c] = nc;
        }
      }
      if (shift <= tol) break;
    }
    var inertia = 0.0;
    for (var i = 0; i < n; i++) {
      inertia += _d2(data, i, centers, _nearest(data, i, centers));
    }
    if (inertia < bestInertia) {
      bestInertia = inertia;
      best = centers;
    }
  }
  return best!;
}

/// k-means++ starting points.
Float64List _plusPlus(Float32List data, int n, _Rng rng) {
  final centers = Float64List(kClusters * 3);
  final first = rng.nextInt(n);
  for (var c = 0; c < 3; c++) {
    centers[c] = data[first * 3 + c];
  }
  final dist = Float64List(n);
  for (var i = 0; i < n; i++) {
    dist[i] = _d2(data, i, centers, 0);
  }
  for (var k = 1; k < kClusters; k++) {
    var sum = 0.0;
    for (final d in dist) {
      sum += d;
    }
    var target = rng.next() * sum, pick = n - 1;
    for (var i = 0; i < n; i++) {
      target -= dist[i];
      if (target <= 0) {
        pick = i;
        break;
      }
    }
    for (var c = 0; c < 3; c++) {
      centers[k * 3 + c] = data[pick * 3 + c];
    }
    for (var i = 0; i < n; i++) {
      final d = _d2(data, i, centers, k);
      if (d < dist[i]) dist[i] = d;
    }
  }
  return centers;
}

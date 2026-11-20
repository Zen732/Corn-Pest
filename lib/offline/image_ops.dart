// Image steps used by the features, written to give exactly the same pixels as OpenCV 5
// (cv2.resize INTER_AREA, cvtColor RGB2GRAY and RGB2HSV for 8-bit images), which
// ml/src/features.py uses in training. Checked pixel for pixel against OpenCV
// (see test/offline_parity_test.dart).
import 'dart:math' as math;
import 'dart:typed_data';

/// An 8-bit image: [channels] = 3 (RGB) or 1 (gray), rows stored one after another.
class Img {
  Img(this.width, this.height, this.channels, [Uint8List? data])
      : data = data ?? Uint8List(width * height * channels);
  final int width, height, channels;
  final Uint8List data;
}

final _f32 = Float32List(1);

/// Rounds like C's float arithmetic (to float32).
double f32(double x) {
  _f32[0] = x;
  return _f32[0];
}

/// cvRound: nearest whole number, ties to even (like C's lrint).
int roundHalfEven(double x) {
  final r = x.floorToDouble();
  final diff = x - r;
  if (diff > 0.5) return r.toInt() + 1;
  if (diff < 0.5) return r.toInt();
  final i = r.toInt();
  return i.isEven ? i : i + 1;
}

int _sat8(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

/// cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).
Img resizeArea(Img src, int w, int h) {
  if (src.width == w && src.height == h) {
    return Img(w, h, src.channels, Uint8List.fromList(src.data));
  }
  final scaleX = src.width / w, scaleY = src.height / h;
  if (scaleX >= 1 && scaleY >= 1) {
    final ix = scaleX.round(), iy = scaleY.round();
    const eps = 2.220446049250313e-16;
    if ((scaleX - ix).abs() < eps && (scaleY - iy).abs() < eps) return _areaFast(src, ix, iy, w, h);
    return _areaGeneral(src, w, h);
  }
  return _linearAreaMode(src, w, h);
}

/// Whole-number shrink: average each ix-by-iy block.
Img _areaFast(Img src, int ix, int iy, int w, int h) {
  final cn = src.channels, out = Img(w, h, cn);
  final scale = f32(1.0 / (ix * iy));
  final rowLen = src.width * cn;
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      for (var c = 0; c < cn; c++) {
        var sum = 0;
        for (var dy = 0; dy < iy; dy++) {
          final base = (y * iy + dy) * rowLen + (x * ix) * cn + c;
          for (var dx = 0; dx < ix; dx++) {
            sum += src.data[base + dx * cn];
          }
        }
        out.data[(y * w + x) * cn + c] =
            (ix == 2 && iy == 2) ? (sum + 2) >> 2 : _sat8(roundHalfEven(f32(sum * scale)));
      }
    }
  }
  return out;
}

class _Tab {
  _Tab(this.di, this.si, this.alpha);
  final int di, si;
  final double alpha; // float32 value
}

List<_Tab> _areaTab(int ssize, int dsize, int cn, double scale) {
  final tab = <_Tab>[];
  for (var dx = 0; dx < dsize; dx++) {
    final fsx1 = dx * scale;
    final fsx2 = fsx1 + scale;
    final cell = math.min(scale, ssize - fsx1);
    var sx1 = fsx1.ceil(), sx2 = fsx2.floor();
    sx2 = math.min(sx2, ssize - 1);
    sx1 = math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.add(_Tab(dx * cn, (sx1 - 1) * cn, f32((sx1 - fsx1) / cell)));
    for (var sx = sx1; sx < sx2; sx++) {
      tab.add(_Tab(dx * cn, sx * cn, f32(1.0 / cell)));
    }
    if (fsx2 - sx2 > 1e-3) {
      tab.add(_Tab(dx * cn, sx2 * cn, f32(math.min(math.min(fsx2 - sx2, 1.0), cell) / cell)));
    }
  }
  return tab;
}

/// Shrink by a non-whole factor: OpenCV's area tables, float32 arithmetic in the same order.
Img _areaGeneral(Img src, int w, int h) {
  final cn = src.channels, width = w * cn, rowLen = src.width * cn;
  final xtab = _areaTab(src.width, w, cn, src.width / w);
  final ytab = _areaTab(src.height, h, 1, src.height / h);
  final out = Img(w, h, cn);
  final buf = Float32List(width), sum = Float32List(width);
  var prevDy = ytab.first.di;
  void write(int dy) {
    for (var i = 0; i < width; i++) {
      out.data[dy * width + i] = _sat8(roundHalfEven(sum[i]));
    }
  }

  for (final t in ytab) {
    final rowStart = t.si * rowLen;
    buf.fillRange(0, width, 0);
    for (final x in xtab) {
      for (var c = 0; c < cn; c++) {
        buf[x.di + c] = buf[x.di + c] + f32(src.data[rowStart + x.si + c] * x.alpha);
      }
    }
    if (t.di != prevDy) {
      write(prevDy);
      for (var i = 0; i < width; i++) {
        sum[i] = t.alpha * buf[i];
      }
      prevDy = t.di;
    } else {
      for (var i = 0; i < width; i++) {
        sum[i] = sum[i] + f32(t.alpha * buf[i]);
      }
    }
  }
  write(prevDy);
  return out;
}

/// INTER_AREA when not shrinking both ways: OpenCV's fixed-point linear resize with
/// "area" coefficients.
Img _linearAreaMode(Img src, int w, int h) {
  final cn = src.channels;
  ({List<int> ofs, List<int> a0, List<int> a1}) coeffs(int ssize, int dsize) {
    final inv = dsize / ssize, scale = 1.0 / inv;
    final ofs = <int>[], a0 = <int>[], a1 = <int>[];
    for (var d = 0; d < dsize; d++) {
      var s = (d * scale).floor();
      var f = f32((d + 1) - (s + 1) * inv);
      f = f <= 0 ? 0.0 : f32(f - f.floorToDouble());
      if (s < 0) {
        f = 0;
        s = 0;
      }
      if (s + 1 >= ssize && s >= ssize - 1) {
        f = 0;
        s = ssize - 1;
      }
      ofs.add(s);
      a0.add(roundHalfEven(f32(1 - f) * 2048));
      a1.add(roundHalfEven(f * 2048));
    }
    return (ofs: ofs, a0: a0, a1: a1);
  }

  final x = coeffs(src.width, w), y = coeffs(src.height, h);
  final rows = <int, Int32List>{};
  Int32List hrow(int sy) => rows.putIfAbsent(sy, () {
        final r = Int32List(w * cn);
        final base = sy * src.width * cn;
        for (var d = 0; d < w; d++) {
          final s = x.ofs[d], s1 = math.min(s + 1, src.width - 1);
          for (var c = 0; c < cn; c++) {
            r[d * cn + c] = src.data[base + s * cn + c] * x.a0[d] + src.data[base + s1 * cn + c] * x.a1[d];
          }
        }
        return r;
      });
  final out = Img(w, h, cn);
  for (var d = 0; d < h; d++) {
    final s = y.ofs[d], s1 = math.min(s + 1, src.height - 1);
    final r0 = hrow(s), r1 = hrow(s1);
    final b0 = y.a0[d], b1 = y.a1[d];
    for (var i = 0; i < w * cn; i++) {
      final v = ((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16);
      out.data[d * w * cn + i] = _sat8((v + 2) >> 2);
    }
  }
  return out;
}

/// cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) in OpenCV 5.
Img toGray(Img rgb) {
  final n = rgb.width * rgb.height, out = Img(rgb.width, rgb.height, 1);
  for (var i = 0; i < n; i++) {
    final r = rgb.data[i * 3], g = rgb.data[i * 3 + 1], b = rgb.data[i * 3 + 2];
    out.data[i] = (r * 9798 + g * 19235 + b * 3735 + (1 << 14)) >> 15;
  }
  return out;
}

final List<int> _sdiv = [0, for (var i = 1; i < 256; i++) roundHalfEven((255 << 12) / i)];
final List<int> _hdiv = [0, for (var i = 1; i < 256; i++) roundHalfEven((180 << 12) / (6.0 * i))];

/// cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV): H 0-179, S and V 0-255.
Img toHsv(Img rgb) {
  final n = rgb.width * rgb.height, out = Img(rgb.width, rgb.height, 3);
  for (var i = 0; i < n; i++) {
    final r = rgb.data[i * 3], g = rgb.data[i * 3 + 1], b = rgb.data[i * 3 + 2];
    final v = math.max(r, math.max(g, b)), vmin = math.min(r, math.min(g, b)), diff = v - vmin;
    final vr = v == r ? -1 : 0, vg = v == g ? -1 : 0;
    final s = (diff * _sdiv[v] + (1 << 11)) >> 12;
    var h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h = (h * _hdiv[diff] + (1 << 11)) >> 12;
    if (h < 0) h += 180;
    out.data[i * 3] = h;
    out.data[i * 3 + 1] = s;
    out.data[i * 3 + 2] = v;
  }
  return out;
}

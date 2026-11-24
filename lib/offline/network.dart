// Version 2: the ready-made image network (EfficientNet-B2 since 25 Nov 2026, MobileNetV3-Large
// before; ImageNet weights) run on the phone in plain Dart, so no extra plugin is needed. It turns
// a photo into a list of numbers (1408 for B2); the four SVMs then decide on those numbers. The
// layers and numbers come from ml/src/network_export.py (assets/models/network.json +
// network.bin); the steps are explained at the top of that file.
//
// The image is kept as height x width x channels, and four channels are worked on at once
// (Float32x4), which the phone's processor does in one step. The two steps that take most of the
// time are written for speed (24 Nov 2026, B2 is about 5 times the work of MobileNetV3): the 1 x 1
// convolution does 4 pixels at a time, so each weight is read once for 4 pixels, and both it and
// the depthwise convolution keep their running sums in local variables (processor registers)
// instead of in memory. The sums are added in the same order as before, so the numbers are the same.
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'image_ops.dart';

class _Tensor {
  _Tensor(this.h, this.w, this.c) : data = Float32List(h * w * c);
  final int h, w, c;
  final Float32List data;
  Float32x4List get v => Float32x4List.view(data.buffer, data.offsetInBytes, data.length ~/ 4);
}

// Activations, as numbers so the inner loops don't compare strings.
const _none = 0, _relu = 1, _hswish = 2, _silu = 3;

int _actCode(String? a) {
  switch (a) {
    case 'relu':
      return _relu;
    case 'hswish':
      return _hswish;
    case 'silu':
      return _silu;
    case 'none':
    case null:
      return _none;
    default:
      throw StateError('unknown activation $a');
  }
}

class _Op {
  _Op(this.kind, {this.inC = 0, this.outC = 0, this.k = 0, this.stride = 1, this.depthwise = false,
      this.act = _none, this.sigmoidGate = false, this.w, this.wt, this.b, this.w2, this.b2, this.squeeze = 0});
  final String kind;
  final int inC, outC, k, stride, squeeze, act;
  final bool depthwise, sigmoidGate;
  final Float32List? w, b, w2, b2;
  final Float32x4List? wt; // 1 x 1 convolution: weights as [out / 4][in], 4 output channels each
}

final _zero = Float32x4.zero();
final _three = Float32x4.splat(3);
final _six = Float32x4.splat(6);
final _sixth = Float32x4.splat(1 / 6);

/// 1 / (1 + e^-v), the logistic sigmoid (EfficientNet).
double _sigmoid(double v) => 1 / (1 + math.exp(-v));

/// The layer's activation on every number of [t], in place.
void _applyAct(_Tensor t, int act) {
  switch (act) {
    case _relu:
      final v = t.v;
      for (var i = 0; i < v.length; i++) {
        v[i] = v[i].max(_zero);
      }
    case _hswish:
      final v = t.v;
      for (var i = 0; i < v.length; i++) {
        final x = v[i];
        v[i] = x * (x + _three).clamp(_zero, _six) * _sixth;
      }
    case _silu: // x * sigmoid(x), one number at a time (there is no four-at-a-time e^x)
      final d = t.data;
      for (var i = 0; i < d.length; i++) {
        final x = d[i];
        d[i] = x * _sigmoid(x);
      }
  }
}

/// [w] stored [in][out] -> [out / 4][in], each element holding 4 output channels.
Float32x4List _transpose(Float32List w, int inC, int outC) {
  final o4 = outC ~/ 4;
  final t = Float32x4List(o4 * inC);
  for (var j = 0; j < o4; j++) {
    for (var ci = 0; ci < inC; ci++) {
      final s = ci * outC + 4 * j;
      t[j * inC + ci] = Float32x4(w[s], w[s + 1], w[s + 2], w[s + 3]);
    }
  }
  return t;
}

class Network {
  Network._(this._ops, this._mean, this._std, this.size, this.length);

  /// [spec] is network.json, [bin] network.bin.
  factory Network.fromBytes(String spec, ByteData bin) {
    final m = jsonDecode(spec) as Map<String, dynamic>;
    final floats = m['floats'] as int;
    final all = Float32List(floats);
    for (var i = 0; i < floats; i++) {
      all[i] = bin.getFloat32(i * 4, Endian.little);
    }
    // Each layer gets its own copy, so the four-at-a-time views line up in memory.
    Float32List part(int start, int n) => Float32List.fromList(Float32List.sublistView(all, start, start + n));
    final ops = <_Op>[];
    for (final o in (m['ops'] as List).cast<Map<String, dynamic>>()) {
      switch (o['op'] as String) {
        case 'conv':
          final inC = o['in'] as int, outC = o['out'] as int, k = o['k'] as int, stride = o['stride'] as int;
          final dw = (o['groups'] as int) != 1;
          if (outC % 4 != 0 || (dw && inC % 4 != 0)) throw StateError('channels must be a multiple of 4');
          final w = part(o['w'] as int, dw ? k * k * inC : k * k * inC * outC);
          final pointwise = !dw && k == 1 && stride == 1;
          ops.add(_Op('conv', inC: inC, outC: outC, k: k, stride: stride, depthwise: dw,
              act: _actCode(o['act'] as String?), w: pointwise ? null : w,
              wt: pointwise ? _transpose(w, inC, outC) : null, b: part(o['b'] as int, outC)));
        case 'se':
          final c = o['c'] as int, s = o['s'] as int;
          if (c % 4 != 0) throw StateError('channels must be a multiple of 4');
          ops.add(_Op('se', inC: c, squeeze: s, act: _actCode((o['act'] as String?) ?? 'relu'),
              sigmoidGate: o['gate'] == 'sigmoid', w: part(o['w1'] as int, c * s), b: part(o['b1'] as int, s),
              w2: part(o['w2'] as int, s * c), b2: part(o['b2'] as int, c)));
        default:
          ops.add(_Op(o['op'] as String));
      }
    }
    List<double> d(String k) => (m[k] as List).map((v) => (v as num).toDouble()).toList();
    return Network._(ops, d('mean'), d('std'), (m['input'] as List)[0] as int, m['output'] as int);
  }

  final List<_Op> _ops;
  final List<double> _mean, _std;
  final int size, length;

  /// The photo's numbers ([length] of them), like NumpyNetwork.vector in ml/src/network.py.
  Float32List vector(Img rgb) {
    var x = _prepare(rgb);
    _Tensor? saved;
    for (final op in _ops) {
      switch (op.kind) {
        case 'conv':
          x = op.depthwise ? _depthwise(x, op) : (op.wt != null ? _pointwise(x, op) : _conv(x, op));
        case 'se':
          _squeezeExcite(x, op);
        case 'save':
          saved = x;
        case 'add':
          final a = x.v, b = saved!.v;
          for (var i = 0; i < a.length; i++) {
            a[i] = a[i] + b[i];
          }
        case 'pool':
          return _mean4(x);
      }
    }
    throw StateError('network.json has no pool step');
  }

  /// Resize to size x size (OpenCV INTER_AREA), 0-1, minus the ImageNet mean, divided by its std;
  /// float32 at every step like numpy.
  _Tensor _prepare(Img rgb) {
    final small = resizeArea(rgb, size, size);
    final t = _Tensor(size, size, 3);
    for (var i = 0; i < small.data.length; i++) {
      final ch = i % 3;
      t.data[i] = f32(f32(f32(small.data[i] / 255.0) - f32(_mean[ch])) / f32(_std[ch]));
    }
    return t;
  }

  static int _outSize(int n, int k, int s) => (n + 2 * ((k - 1) ~/ 2) - k) ~/ s + 1;

  /// Ordinary convolution (the first layer only): every output channel sees every input channel.
  _Tensor _conv(_Tensor x, _Op op) {
    final k = op.k, s = op.stride, p = (k - 1) ~/ 2;
    final oh = _outSize(x.h, k, s), ow = _outSize(x.w, k, s);
    final out = _Tensor(oh, ow, op.outC);
    final o4 = op.outC ~/ 4;
    final w = Float32x4List.view(op.w!.buffer), b = Float32x4List.view(op.b!.buffer), ov = out.v;
    for (var oy = 0; oy < oh; oy++) {
      for (var ox = 0; ox < ow; ox++) {
        final base = (oy * ow + ox) * o4;
        for (var j = 0; j < o4; j++) {
          ov[base + j] = b[j];
        }
        for (var ky = 0; ky < k; ky++) {
          final iy = oy * s - p + ky;
          if (iy < 0 || iy >= x.h) continue;
          for (var kx = 0; kx < k; kx++) {
            final ix = ox * s - p + kx;
            if (ix < 0 || ix >= x.w) continue;
            for (var ci = 0; ci < x.c; ci++) {
              final v = Float32x4.splat(x.data[(iy * x.w + ix) * x.c + ci]);
              final wb = ((ky * k + kx) * x.c + ci) * o4;
              for (var j = 0; j < o4; j++) {
                ov[base + j] = ov[base + j] + v * w[wb + j];
              }
            }
          }
        }
      }
    }
    _applyAct(out, op.act);
    return out;
  }

  /// 1 x 1 convolution: each pixel's channels times a matrix. Four pixels at a time: their values
  /// are copied once into all four lanes ([xs]), then every 4 output channels are summed over the
  /// input channels in local variables.
  _Tensor _pointwise(_Tensor x, _Op op) {
    final n = x.h * x.w, inC = x.c, o4 = op.outC ~/ 4;
    final out = _Tensor(x.h, x.w, op.outC);
    final wt = op.wt!, b = Float32x4List.view(op.b!.buffer), ov = out.v, xd = x.data;
    final xs = Float32x4List(4 * inC); // [ci][pixel]
    for (var p0 = 0; p0 < n; p0 += 4) {
      final m = n - p0 < 4 ? n - p0 : 4; // pixels in this group (the last one may have fewer)
      for (var t = 0; t < 4; t++) {
        final src = (p0 + (t < m ? t : 0)) * inC;
        for (var ci = 0; ci < inC; ci++) {
          xs[ci * 4 + t] = Float32x4.splat(xd[src + ci]);
        }
      }
      // 8 output channels (two groups of 4) for the 4 pixels: each value read is used twice.
      var j = 0;
      for (; j + 2 <= o4; j += 2) {
        final bj = b[j], bk = b[j + 1];
        var a0 = bj, a1 = bj, a2 = bj, a3 = bj, c0 = bk, c1 = bk, c2 = bk, c3 = bk;
        final wb = j * inC, wc = wb + inC;
        for (var ci = 0; ci < inC; ci++) {
          final wv = wt[wb + ci], wu = wt[wc + ci];
          final xb = ci * 4;
          final x0 = xs[xb], x1 = xs[xb + 1], x2 = xs[xb + 2], x3 = xs[xb + 3];
          a0 = a0 + x0 * wv;
          a1 = a1 + x1 * wv;
          a2 = a2 + x2 * wv;
          a3 = a3 + x3 * wv;
          c0 = c0 + x0 * wu;
          c1 = c1 + x1 * wu;
          c2 = c2 + x2 * wu;
          c3 = c3 + x3 * wu;
        }
        final ob = p0 * o4 + j;
        ov[ob] = a0;
        ov[ob + 1] = c0;
        if (m > 1) {
          ov[ob + o4] = a1;
          ov[ob + o4 + 1] = c1;
        }
        if (m > 2) {
          ov[ob + 2 * o4] = a2;
          ov[ob + 2 * o4 + 1] = c2;
        }
        if (m > 3) {
          ov[ob + 3 * o4] = a3;
          ov[ob + 3 * o4 + 1] = c3;
        }
      }
      for (; j < o4; j++) {
        final bj = b[j];
        var a0 = bj, a1 = bj, a2 = bj, a3 = bj;
        final wb = j * inC;
        for (var ci = 0; ci < inC; ci++) {
          final wv = wt[wb + ci];
          final xb = ci * 4;
          a0 = a0 + xs[xb] * wv;
          a1 = a1 + xs[xb + 1] * wv;
          a2 = a2 + xs[xb + 2] * wv;
          a3 = a3 + xs[xb + 3] * wv;
        }
        final ob = p0 * o4 + j;
        ov[ob] = a0;
        if (m > 1) ov[ob + o4] = a1;
        if (m > 2) ov[ob + 2 * o4] = a2;
        if (m > 3) ov[ob + 3 * o4] = a3;
      }
    }
    _applyAct(out, op.act);
    return out;
  }

  /// k x k convolution of each channel on its own. For each output pixel the taps inside the
  /// image are listed once, then 16 channels at a time are summed in local variables.
  _Tensor _depthwise(_Tensor x, _Op op) {
    final k = op.k, s = op.stride, p = (k - 1) ~/ 2;
    final oh = _outSize(x.h, k, s), ow = _outSize(x.w, k, s);
    final out = _Tensor(oh, ow, x.c);
    final c4 = x.c ~/ 4;
    final w = Float32x4List.view(op.w!.buffer), b = Float32x4List.view(op.b!.buffer), ov = out.v, xv = x.v;
    final taps = Int32List(2 * k * k); // (input pixel, weight) start of each tap, in ky, kx order
    for (var oy = 0; oy < oh; oy++) {
      for (var ox = 0; ox < ow; ox++) {
        var nt = 0;
        for (var ky = 0; ky < k; ky++) {
          final iy = oy * s - p + ky;
          if (iy < 0 || iy >= x.h) continue;
          for (var kx = 0; kx < k; kx++) {
            final ix = ox * s - p + kx;
            if (ix < 0 || ix >= x.w) continue;
            taps[nt++] = (iy * x.w + ix) * c4;
            taps[nt++] = (ky * k + kx) * c4;
          }
        }
        final ob = (oy * ow + ox) * c4;
        var j = 0;
        for (; j + 4 <= c4; j += 4) {
          var a0 = b[j], a1 = b[j + 1], a2 = b[j + 2], a3 = b[j + 3];
          for (var t = 0; t < nt; t += 2) {
            final ib = taps[t] + j, wb = taps[t + 1] + j;
            a0 = a0 + xv[ib] * w[wb];
            a1 = a1 + xv[ib + 1] * w[wb + 1];
            a2 = a2 + xv[ib + 2] * w[wb + 2];
            a3 = a3 + xv[ib + 3] * w[wb + 3];
          }
          ov[ob + j] = a0;
          ov[ob + j + 1] = a1;
          ov[ob + j + 2] = a2;
          ov[ob + j + 3] = a3;
        }
        for (; j < c4; j++) {
          var a = b[j];
          for (var t = 0; t < nt; t += 2) {
            a = a + xv[taps[t] + j] * w[taps[t + 1] + j];
          }
          ov[ob + j] = a;
        }
      }
    }
    _applyAct(out, op.act);
    return out;
  }

  /// Average of every channel over the image.
  static Float32List _mean4(_Tensor x) {
    final c4 = x.c ~/ 4, xv = x.v;
    final sum = Float32x4List(c4);
    for (var px = 0; px < x.h * x.w; px++) {
      for (var j = 0; j < c4; j++) {
        sum[j] = sum[j] + xv[px * c4 + j];
      }
    }
    final n = Float32x4.splat(1 / (x.h * x.w));
    for (var j = 0; j < c4; j++) {
      sum[j] = sum[j] * n;
    }
    return Float32List.view(sum.buffer);
  }

  /// Squeeze-excitation: average over the image -> small layer (relu or silu) -> layer back to
  /// the channels (hard sigmoid or sigmoid) -> each channel multiplied by its number.
  void _squeezeExcite(_Tensor x, _Op op) {
    final c = op.inC, sq = op.squeeze;
    final m = _mean4(x);
    final hidden = Float32List.fromList(op.b!);
    for (var ci = 0; ci < c; ci++) {
      final v = m[ci];
      for (var j = 0; j < sq; j++) {
        hidden[j] += v * op.w![ci * sq + j];
      }
    }
    final scale = Float32List.fromList(op.b2!);
    final silu = op.act == _silu;
    for (var j = 0; j < sq; j++) {
      final h = hidden[j];
      final v = silu ? h * _sigmoid(h) : (h > 0 ? h : 0.0);
      if (v == 0) continue;
      for (var ci = 0; ci < c; ci++) {
        scale[ci] += v * op.w2![j * c + ci];
      }
    }
    for (var ci = 0; ci < c; ci++) {
      scale[ci] = op.sigmoidGate ? _sigmoid(scale[ci]) : (scale[ci] + 3).clamp(0, 6) / 6;
    }
    final c4 = c ~/ 4, xv = x.v, sv = Float32x4List.view(scale.buffer);
    for (var px = 0; px < x.h * x.w; px++) {
      for (var j = 0; j < c4; j++) {
        xv[px * c4 + j] = xv[px * c4 + j] * sv[j];
      }
    }
  }
}

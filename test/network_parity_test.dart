// Does the phone's image network (lib/offline/network.dart) give PyTorch's numbers?
// Needs build/parity/network/ (ml/src/network_export.py) and assets/models/network.*, which
// stay on the training PC; elsewhere this test is skipped.
//     flutter test test/network_parity_test.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/offline/image_ops.dart';
import 'package:pest_assessment/offline/network.dart';

void main() {
  final dir = Directory('build/parity/network');
  final skip = !File('${dir.path}/refs.json').existsSync() || !File('assets/models/network.bin').existsSync()
      ? 'reference files are only on the training PC'
      : null;

  test('same pixels in: Dart network matches PyTorch', () {
    final net = Network.fromBytes(File('assets/models/network.json').readAsStringSync(),
        ByteData.sublistView(File('assets/models/network.bin').readAsBytesSync()));
    final refs = (jsonDecode(File('${dir.path}/refs.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();
    var worst = 0.0;
    final timer = Stopwatch()..start();
    for (final r in refs) {
      final img = Img(r['width'] as int, r['height'] as int, 3, File('${dir.path}/${r['file']}').readAsBytesSync());
      final v = net.vector(img);
      final ref = (r['vector'] as List).cast<num>();
      expect(v.length, ref.length);
      var top = 0.0, diff = 0.0;
      for (var i = 0; i < v.length; i++) {
        top = ref[i].abs() > top ? ref[i].abs().toDouble() : top;
        diff = (v[i] - ref[i]).abs() > diff ? (v[i] - ref[i]).abs() : diff;
      }
      if (diff / top > worst) worst = diff / top;
    }
    // ignore: avoid_print
    print('${refs.length} photos, ${(timer.elapsedMilliseconds / refs.length).toStringAsFixed(0)} ms each here; '
        'largest difference ${worst.toStringAsExponential(2)} of the largest value');
    expect(worst, lessThan(1e-4));
  }, skip: skip, timeout: const Timeout(Duration(minutes: 10)));
}

// The offline checker's background worker: one isolate that loads the models and the image
// network once (about 120 MB in Version 2) and then checks photo after photo, so the screen keeps
// moving and the models are not copied for every scan.
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'flow.dart';
import 'image_ops.dart';
import 'network.dart';
import 'svm_model.dart';

class OfflineChecker {
  OfflineChecker._(this._port, this.version);
  final SendPort _port;

  /// The models' version (model_version.txt).
  final String version;

  /// Starts the worker. [texts]: each file's .json ("corn_check", "detector", "pest", "health" and,
  /// if a model uses the network, "network"); [bins]: the matching .bin files.
  static Future<OfflineChecker> start(Map<String, String> texts, Map<String, ByteData> bins, String version) async {
    final ready = ReceivePort();
    final transfer = {
      for (final e in bins.entries)
        e.key: TransferableTypedData.fromList([e.value.buffer.asUint8List(e.value.offsetInBytes, e.value.lengthInBytes)])
    };
    await Isolate.spawn(_main, [ready.sendPort, texts, transfer, version], onError: ready.sendPort);
    final first = await ready.first;
    if (first is SendPort) return OfflineChecker._(first, version);
    throw StateError('offline checker did not start: $first');
  }

  /// The same answer as assessPhoto (flow.dart), worked out in the background.
  Future<Map<String, dynamic>> check(Img img) async {
    final reply = ReceivePort();
    _port.send([reply.sendPort, img.width, img.height, TransferableTypedData.fromList([img.data])]);
    final answer = await reply.first;
    if (answer is Map) return answer.cast<String, dynamic>();
    throw StateError('$answer');
  }
}

void _main(List<Object?> args) {
  final out = args[0] as SendPort;
  final OfflineModels models;
  try {
    final texts = (args[1] as Map).cast<String, String>();
    final bins = (args[2] as Map)
        .map((k, v) => MapEntry(k as String, ByteData.view((v as TransferableTypedData).materialize())));
    SvmModel one(String name) => SvmModel.fromBytes(texts[name]!, bins[name]!);
    models = OfflineModels(
      cornCheck: one('corn_check'),
      detector: one('detector'),
      pest: one('pest'),
      health: one('health'),
      version: args[3] as String,
      network: bins.containsKey('network') ? Network.fromBytes(texts['network']!, bins['network']!) : null,
    );
  } catch (e) {
    out.send('could not load the models: $e');
    return;
  }
  final port = ReceivePort();
  out.send(port.sendPort);
  port.listen((message) {
    final m = message as List;
    final reply = m[0] as SendPort;
    try {
      final img = Img(m[1] as int, m[2] as int, 3, (m[3] as TransferableTypedData).materialize().asUint8List());
      reply.send(assessPhoto(img, models));
    } catch (e) {
      reply.send('check failed: $e');
    }
  });
}

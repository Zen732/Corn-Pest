import 'dart:typed_data';
import 'dart:ui' as ui;

import 'image_ops.dart';

/// Decodes a JPG or PNG into RGB with Flutter's own image decoder (EXIF rotation applied,
/// transparency dropped like OpenCV's IMREAD_COLOR).
Future<Img> decodePhoto(Uint8List bytes) async {
  final codec = await ui.instantiateImageCodec(bytes);
  final frame = await codec.getNextFrame();
  final image = frame.image;
  final rgba = await image.toByteData(format: ui.ImageByteFormat.rawStraightRgba);
  final out = Img(image.width, image.height, 3);
  final px = rgba!.buffer.asUint8List(rgba.offsetInBytes, rgba.lengthInBytes);
  for (var i = 0, j = 0; i < px.length; i += 4, j += 3) {
    out.data[j] = px[i];
    out.data[j + 1] = px[i + 1];
    out.data[j + 2] = px[i + 2];
  }
  image.dispose();
  codec.dispose();
  return out;
}

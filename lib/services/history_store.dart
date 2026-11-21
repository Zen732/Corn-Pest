import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'scan_service.dart';

/// One scan in History. Saved on the phone first, uploaded to Supabase when there is internet.
class LocalScan {
  LocalScan({
    required this.id,
    required this.createdAt,
    required this.result,
    required this.remotePhotoPath,
    this.photoName,
    this.synced = false,
    this.deleted = false,
  });

  final String id; // also the row id in Supabase, so retrying an upload never makes a copy
  final DateTime createdAt; // when the photo was checked (not when it was uploaded)
  final ScanResult result;
  final String remotePhotoPath; // <user id>/<id>.jpg in the scan-photos bucket
  String? photoName; // the photo on this phone; null for scans from another phone not yet downloaded
  bool synced; // uploaded to Supabase
  bool deleted; // deleted here; the deletion still has to reach Supabase

  Map<String, dynamic> toJson() => {
        'id': id,
        'created_at': createdAt.toUtc().toIso8601String(),
        'result': result.toJson(),
        'remote_photo_path': remotePhotoPath,
        'photo_name': photoName,
        'synced': synced,
        'deleted': deleted,
      };

  factory LocalScan.fromJson(Map<String, dynamic> j) => LocalScan(
        id: j['id'] as String,
        createdAt: DateTime.parse(j['created_at'] as String),
        result: ScanResult.fromJson(j['result'] as Map<String, dynamic>),
        remotePhotoPath: j['remote_photo_path'] as String,
        photoName: j['photo_name'] as String?,
        synced: j['synced'] as bool? ?? false,
        deleted: j['deleted'] as bool? ?? false,
      );
}

/// Files kept on the phone (the app's private folder). An interface so tests can use memory.
abstract class BlobStorage {
  Future<void> write(String name, List<int> bytes);
  Future<Uint8List?> read(String name);
  Future<List<String>> names();
  Future<void> delete(String name);
}

class DirStorage implements BlobStorage {
  DirStorage([Future<Directory> Function()? dir]) : _dir = dir ?? _defaultDir;
  final Future<Directory> Function() _dir;
  Directory? _resolved;

  static Future<Directory> _defaultDir() async =>
      Directory('${(await getApplicationDocumentsDirectory()).path}${Platform.pathSeparator}scans');

  Future<Directory> get _d async => _resolved ??= await (await _dir()).create(recursive: true);

  File _file(Directory d, String name) => File('${d.path}${Platform.pathSeparator}$name');

  @override
  Future<void> write(String name, List<int> bytes) async {
    final d = await _d;
    final tmp = _file(d, '$name.tmp');
    await tmp.writeAsBytes(bytes, flush: true);
    await tmp.rename(_file(d, name).path); // a crash never leaves half a file
  }

  @override
  Future<Uint8List?> read(String name) async {
    final f = _file(await _d, name);
    return await f.exists() ? await f.readAsBytes() : null;
  }

  @override
  Future<List<String>> names() async => [
        await for (final e in (await _d).list())
          if (e is File && !e.path.endsWith('.tmp')) e.uri.pathSegments.last,
      ];

  @override
  Future<void> delete(String name) async {
    final f = _file(await _d, name);
    if (await f.exists()) await f.delete();
  }
}

class MemoryStorage implements BlobStorage {
  final files = <String, Uint8List>{};
  @override
  Future<void> write(String name, List<int> bytes) async => files[name] = Uint8List.fromList(bytes);
  @override
  Future<Uint8List?> read(String name) async => files[name];
  @override
  Future<List<String>> names() async => files.keys.toList();
  @override
  Future<void> delete(String name) async => files.remove(name);
}

/// A scan that is in Supabase.
class RemoteScan {
  const RemoteScan({required this.id, required this.createdAt, required this.result, required this.photoPath});
  final String id;
  final DateTime createdAt;
  final ScanResult result;
  final String photoPath;
}

/// The user's scans in Supabase. An interface so tests can fake it (and go "offline").
abstract class ScanRemote {
  String? get userId;

  /// Photo + row. Safe to repeat: an already-uploaded photo or row counts as done.
  Future<void> upload(LocalScan scan, Uint8List photo);
  Future<void> remove(LocalScan scan);
  Future<List<RemoteScan>> list();
  Future<Uint8List> downloadPhoto(String path);
}

class SupabaseScanRemote implements ScanRemote {
  SupabaseClient get _db => Supabase.instance.client;

  @override
  String? get userId => _db.auth.currentUser?.id;

  @override
  Future<void> upload(LocalScan scan, Uint8List photo) async {
    final png = scan.remotePhotoPath.endsWith('.png');
    try {
      await _db.storage.from('scan-photos').uploadBinary(scan.remotePhotoPath, photo,
          fileOptions: FileOptions(contentType: png ? 'image/png' : 'image/jpeg'));
    } on StorageException catch (e) {
      final exists = e.statusCode == '409' || e.message.toLowerCase().contains('exist');
      if (!exists) rethrow; // uploaded by an earlier attempt
    }
    final r = scan.result;
    try {
      await _db.from('scans').insert({
        'id': scan.id,
        'created_at': scan.createdAt.toUtc().toIso8601String(),
        'photo_path': scan.remotePhotoPath,
        'outcome': r.outcome,
        'pest_key': r.outcome == 'pest' ? r.pest : null,
        'pest_confidence': r.outcome == 'pest' ? r.pestConfidence : null,
        'damage_percent': r.damagePercent,
        'severity': r.severity,
        'model_version': r.modelVersion,
        'details': r.details,
        // barangay_id is filled in from the user's profile by the database
      });
    } on PostgrestException catch (e) {
      if (e.code != '23505') rethrow; // 23505 = this id is already saved
    }
  }

  @override
  Future<void> remove(LocalScan scan) async {
    await _db.from('scans').delete().eq('id', scan.id);
    await _db.storage.from('scan-photos').remove([scan.remotePhotoPath]);
  }

  @override
  Future<List<RemoteScan>> list() async {
    final rows = await _db
        .from('scans')
        .select('id, created_at, photo_path, outcome, pest_key, pest_confidence, damage_percent, severity, model_version, details')
        .order('created_at', ascending: false)
        .limit(500);
    return [
      for (final row in rows)
        RemoteScan(
          id: row['id'] as String,
          createdAt: DateTime.parse(row['created_at'] as String),
          photoPath: row['photo_path'] as String,
          result: ScanResult.fromJson({
            ...((row['details'] as Map<String, dynamic>?) ?? const {}),
            'outcome': row['outcome'],
            'pest': row['pest_key'],
            'pest_confidence': row['pest_confidence'],
            'damage_percent': row['damage_percent'],
            'severity': row['severity'],
            'model_version': row['model_version'],
          }),
        ),
    ];
  }

  @override
  Future<Uint8List> downloadPhoto(String path) => _db.storage.from('scan-photos').download(path);
}

class SyncReport {
  const SyncReport({this.uploaded = 0, this.waiting = 0, this.offline = false});
  final int uploaded, waiting;
  final bool offline;
}

/// History: scans are saved on the phone first (works without internet) and uploaded to
/// Supabase whenever sync() can reach it. Each signed-in user has their own files.
class HistoryStore {
  HistoryStore({required this.storage, required this.remote, this.autoSync = true});
  final BlobStorage storage;
  final ScanRemote remote;

  /// Start an upload in the background after each save or delete (tests turn it off to
  /// check each step).
  final bool autoSync;

  /// Ticks whenever the list changes (saved, uploaded, deleted, downloaded).
  final changes = ValueNotifier<int>(0);
  Future<SyncReport>? _syncing, _nextSync;

  String get _uid => remote.userId ?? 'signed-out';
  String _meta(String id) => '${_uid}_$id.json';

  static String newId() {
    final b = List<int>.generate(16, (_) => Random.secure().nextInt(256));
    b[6] = (b[6] & 0x0f) | 0x40; // UUID version 4
    b[8] = (b[8] & 0x3f) | 0x80;
    final h = b.map((x) => x.toRadixString(16).padLeft(2, '0')).join();
    return '${h.substring(0, 8)}-${h.substring(8, 12)}-${h.substring(12, 16)}-${h.substring(16, 20)}-${h.substring(20)}';
  }

  Future<void> _write(LocalScan s) => storage.write(_meta(s.id), utf8.encode(jsonEncode(s.toJson())));

  /// The latest saved copy (a screen's copy may be older than a background upload).
  Future<LocalScan?> _fresh(String id) async {
    final bytes = await storage.read(_meta(id));
    return bytes == null ? null : LocalScan.fromJson(jsonDecode(utf8.decode(bytes)) as Map<String, dynamic>);
  }

  Future<List<LocalScan>> _all() async {
    final out = <LocalScan>[];
    for (final name in await storage.names()) {
      if (!name.startsWith('${_uid}_') || !name.endsWith('.json')) continue;
      final bytes = await storage.read(name);
      if (bytes == null) continue;
      try {
        out.add(LocalScan.fromJson(jsonDecode(utf8.decode(bytes)) as Map<String, dynamic>));
      } catch (_) {
        // a damaged file: skip it rather than break History
      }
    }
    return out;
  }

  /// Saves on the phone right away, then tries to upload in the background.
  Future<LocalScan> save(ScanResult result, Uint8List photo, {DateTime? at}) async {
    final id = newId();
    final ext = (photo.length > 4 && photo[0] == 0x89 && photo[1] == 0x50) ? 'png' : 'jpg';
    final photoName = '${_uid}_$id.$ext';
    await storage.write(photoName, photo);
    final scan = LocalScan(
      id: id,
      createdAt: at ?? DateTime.now(),
      result: result,
      photoName: photoName,
      remotePhotoPath: '$_uid/$id.$ext',
    );
    await _write(scan);
    changes.value++;
    if (autoSync) unawaited(sync());
    return scan;
  }

  /// The user's scans, newest first.
  Future<List<LocalScan>> list() async {
    final all = (await _all()).where((s) => !s.deleted).toList()..sort((a, b) => b.createdAt.compareTo(a.createdAt));
    return all;
  }

  /// The photo, from the phone; a scan from another phone is downloaded once (needs internet).
  Future<Uint8List?> photo(LocalScan scan) async {
    if (scan.photoName != null) {
      final b = await storage.read(scan.photoName!);
      if (b != null) return b;
    }
    try {
      final bytes = await remote.downloadPhoto(scan.remotePhotoPath);
      final ext = scan.remotePhotoPath.endsWith('.png') ? 'png' : 'jpg';
      final name = '${_uid}_${scan.id}.$ext';
      await storage.write(name, bytes);
      scan.photoName = name;
      final latest = await _fresh(scan.id);
      if (latest != null && !latest.deleted) {
        latest.photoName = name;
        await _write(latest);
      }
      return bytes;
    } catch (_) {
      return null; // offline: the screen shows a placeholder
    }
  }

  /// Removes it from History now; the deletion reaches Supabase on the next sync.
  Future<void> delete(LocalScan scan) async {
    final latest = await _fresh(scan.id) ?? scan;
    latest.deleted = true;
    await _write(latest);
    if (latest.photoName != null) await storage.delete(latest.photoName!);
    scan.deleted = true;
    changes.value++;
    if (autoSync) unawaited(sync());
  }

  /// Uploads waiting scans, sends deletions, and adds scans saved from other phones.
  /// Never throws: without internet it just reports how many are still waiting.
  /// A sync asked for while one is running starts a fresh pass right after it (the running
  /// one may have started while offline).
  Future<SyncReport> sync() {
    if (_syncing == null) {
      _syncing = _sync().whenComplete(() => _syncing = null);
      return _syncing!;
    }
    return _nextSync ??= _syncing!.then((_) {
      _nextSync = null;
      return sync();
    });
  }

  Future<SyncReport> _sync() async {
    if (remote.userId == null) return const SyncReport(offline: true);
    var uploaded = 0, waiting = 0, offline = false, changed = false;
    final all = await _all();
    for (final s in all) {
      try {
        if (s.deleted) {
          if (s.synced) await remote.remove(s);
          await storage.delete(_meta(s.id));
          changed = true;
        } else if (!s.synced) {
          final bytes = s.photoName == null ? null : await storage.read(s.photoName!);
          if (bytes == null) continue;
          await remote.upload(s, bytes);
          s.synced = true;
          await _write(s);
          uploaded++;
          changed = true;
        }
      } catch (_) {
        offline = true;
        if (!s.deleted) waiting++;
      }
    }
    if (!offline) {
      try {
        final known = {for (final s in all) s.id};
        for (final r in await remote.list()) {
          if (known.contains(r.id)) continue;
          await _write(LocalScan(
              id: r.id, createdAt: r.createdAt, result: r.result, remotePhotoPath: r.photoPath, synced: true));
          changed = true;
        }
      } catch (_) {
        offline = true;
      }
    }
    if (changed) changes.value++;
    return SyncReport(uploaded: uploaded, waiting: waiting, offline: offline);
  }
}

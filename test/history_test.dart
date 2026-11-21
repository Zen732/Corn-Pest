// Tests for offline History (part 3): saved on the phone first, uploaded when online.
// Fakes stand in for Supabase (with an on/off "internet" switch); no network needed.
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/services/history_store.dart';
import 'package:pest_assessment/services/scan_service.dart';
import 'package:pest_assessment/strings.dart';

import 'fakes.dart';

const pest = ScanResult(outcome: 'pest', pest: 'fall_armyworm', pestConfidence: 0.84, severity: 'Moderate', damagePercent: 23.4);
const healthy = ScanResult(outcome: 'healthy', severity: 'Healthy', damagePercent: 1.0);

void main() {
  group('HistoryStore', () {
    test('offline: saved on the phone, waiting; back online: uploaded once', () async {
      final remote = FakeRemote()..online = false;
      final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
      final scan = await history.save(pest, tinyPng);
      var report = await history.sync();
      expect(report.offline, isTrue);
      expect(report.waiting, 1);
      expect((await history.list()).single.synced, isFalse);
      expect(remote.rows, isEmpty);

      remote.online = true;
      report = await history.sync();
      expect(report.uploaded, 1);
      expect(remote.rows.keys, [scan.id]);
      expect((await history.list()).single.synced, isTrue);
      report = await history.sync(); // nothing left to upload; no copies
      expect(report.uploaded, 0);
      expect(remote.rows, hasLength(1));
    });

    test('a scan keeps its time and result through saving and reloading', () async {
      final storage = MemoryStorage();
      final at = DateTime.utc(2026, 11, 22, 8, 30);
      await HistoryStore(storage: storage, remote: FakeRemote()..online = false).save(pest, tinyPng, at: at);
      final reloaded = (await HistoryStore(storage: storage, remote: FakeRemote()).list()).single;
      expect(reloaded.createdAt, at);
      expect(reloaded.result.pest, 'fall_armyworm');
      expect(reloaded.result.damagePercent, 23.4);
      expect(reloaded.remotePhotoPath, startsWith('user-1/'));
    });

    test('newest first; deleting offline reaches Supabase later', () async {
      final remote = FakeRemote();
      final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
      final older = await history.save(healthy, tinyPng, at: DateTime(2026, 11, 1));
      final newer = await history.save(pest, tinyPng, at: DateTime(2026, 11, 2));
      await history.sync();
      expect((await history.list()).map((s) => s.id), [newer.id, older.id]);

      remote.online = false;
      await history.delete(older);
      expect((await history.list()).map((s) => s.id), [newer.id]); // gone from History right away
      expect(remote.rows.containsKey(older.id), isTrue); // but still in Supabase while offline
      remote.online = true;
      await history.sync();
      expect(remote.removed, [older.id]);
      expect(remote.rows.keys, [newer.id]);
    });

    test('deleting a scan that never uploaded doesn\'t touch Supabase', () async {
      final remote = FakeRemote()..online = false;
      final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
      final scan = await history.save(pest, tinyPng);
      await history.delete(scan);
      remote.online = true;
      await history.sync();
      expect(remote.removed, isEmpty);
      expect(remote.rows, isEmpty);
    });

    test('scans saved on another phone show up, and their photo is downloaded once', () async {
      final remote = FakeRemote()
        ..fromOtherPhone.add(RemoteScan(id: 'other-1', createdAt: DateTime(2026, 10, 1), result: healthy, photoPath: 'user-1/other-1.jpg'));
      final storage = MemoryStorage();
      final history = HistoryStore(storage: storage, remote: remote);
      await history.sync();
      final scan = (await history.list()).single;
      expect(scan.id, 'other-1');
      expect(scan.synced, isTrue);
      expect(await history.photo(scan), isNotNull);
      remote.online = false;
      expect(await history.photo(scan), isNotNull, reason: 'kept on the phone after the first download');
    });

    test('each account sees only its own scans on a shared phone', () async {
      final storage = MemoryStorage();
      final remote = FakeRemote()..online = false;
      final history = HistoryStore(storage: storage, remote: remote);
      await history.save(pest, tinyPng);
      remote.userId = 'user-2';
      expect(await history.list(), isEmpty);
    });

    test('an upload that stopped halfway is simply retried', () async {
      final remote = _HalfwayRemote();
      final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
      await history.save(pest, tinyPng);
      expect((await history.sync()).waiting, 1);
      expect((await history.sync()).uploaded, 1);
      expect(remote.rows, hasLength(1));
    });
  });

  test('deleting with an older copy of the scan still deletes it from Supabase', () async {
    final remote = FakeRemote();
    final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
    final scan = await history.save(pest, tinyPng); // this copy still says "not uploaded"
    await history.sync(); // uploaded in the background meanwhile
    await history.delete(scan);
    await history.sync();
    expect(remote.removed, [scan.id]);
    expect(remote.rows, isEmpty);
  });

  test('files on the phone: write, read, list, delete', () async {
    final dir = await Directory.systemTemp.createTemp('history_test');
    addTearDown(() => dir.delete(recursive: true));
    final storage = DirStorage(() async => dir);
    await storage.write('a.json', [1, 2, 3]);
    expect(await storage.read('a.json'), [1, 2, 3]);
    expect(await storage.names(), ['a.json']);
    await storage.delete('a.json');
    expect(await storage.read('a.json'), isNull);
    expect(await storage.names(), isEmpty);
  });

  testWidgets('History screen: list, waiting note, details, delete', (tester) async {
    final remote = FakeRemote()..online = false;
    final history = HistoryStore(storage: MemoryStorage(), remote: remote);
    await history.save(healthy, tinyPng, at: DateTime(2026, 11, 1, 9));
    await history.save(pest, tinyPng, at: DateTime(2026, 11, 2, 9));
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), history: history));

    await tester.tap(find.text(S.history));
    await tester.pumpAndSettle();
    expect(find.text(S.waitingCount(2)), findsOneWidget);
    expect(find.text(S.summary('pest', 'fall_armyworm', 'Moderate')), findsOneWidget);
    expect(find.text(S.healthyTitle), findsOneWidget);

    await tester.tap(find.text(S.summary('pest', 'fall_armyworm', 'Moderate')));
    await tester.pumpAndSettle();
    expect(find.text('Fall armyworm'), findsOneWidget);
    expect(find.text(S.waitingNote), findsOneWidget);
    await tester.tap(find.text(S.deleteScan));
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(FilledButton, S.delete));
    await tester.pumpAndSettle();
    expect(find.text(S.summary('pest', 'fall_armyworm', 'Moderate')), findsNothing);
    expect(find.text(S.waitingCount(1)), findsOneWidget);

    remote.online = true;
    await tester.fling(find.text(S.healthyTitle), const Offset(0, 400), 1000); // pull to refresh
    await tester.pumpAndSettle();
    expect(find.textContaining('waiting to upload'), findsNothing);
    expect(remote.rows, hasLength(1));
  });
}

/// Uploads the photo but loses the connection before the row, the first time.
class _HalfwayRemote extends FakeRemote {
  var calls = 0;
  @override
  Future<void> upload(LocalScan scan, photo) async {
    if (calls++ == 0) throw Exception('connection lost');
    return super.upload(scan, photo);
  }
}

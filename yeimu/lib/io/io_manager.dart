import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:yeimu/io/sensor_data_file.dart';
import 'dart:io';
import 'dart:convert';

import 'package:yeimu/versioning/version.dart';

import '../structure/sensor_reading.dart';
import '../structure/timestamps.dart';

class IOManager {
  static Future<bool> saveData(
      String? filename, List<SensorReading> data, List<Timestamp>? timestamps) async {
    List<String> files = await listCompatibleFilenames();
    final File file = await _getLocalFile(filename ?? 'data_${files.length}');
    return await SensorDataFile.save(file, timestamps ?? [], data);
  }

  static Future<SensorDataFile> loadData(String filename) async {
    // note: no compatibility check, assumes the file has been picked from the list provided by listCompatibleFiles()
    final File file = await _getLocalFile(filename);
    final String fileContents = await file.readAsString();
    return SensorDataFile.from(filename, fileContents);
  }

  static Future<List<File>> listCompatibleFiles() async {
    final Directory dir = await _localDirectory;
    return dir.listSync().whereType<File>().toList();
  }

  static Future<List<String>> listCompatibleFilenames() async {
    List<File> files = await listCompatibleFiles();
    final List<String> compatibleFileNames = [];
    for (final File file in files) {
      if (await _isDataFileCompatible(file)) {
        compatibleFileNames.add(file.path.split('/').last.split('.').first);
      }
    }
    return compatibleFileNames;
  }

  static Future<bool> _isDataFileCompatible(File file) async {
    try {
      if (file.path.substring(file.path.length - 5) != '.eimu') {
        return false;
      }
      final String firstLine =
          await file.openRead().transform(utf8.decoder).transform(const LineSplitter()).first;
      return int.parse(firstLine) >= Version.lowestCompatibleVersion;
    } catch (e) {
      print("exception parsing file: $file");
      return false;
    }
  }

  static Future<Directory> get _localDirectory async {
    return await getApplicationDocumentsDirectory();
  }

  static Future<File> _getLocalFile(String filename) async {
    final Directory dir = await _localDirectory;
    return File('${dir.path}/$filename.eimu');
  }

  static Future<void> shareAllFiles() async {
    final List<File> files = await listCompatibleFiles();
    Share.shareXFiles(files.map((f) => XFile(f.path)).toList());
  }

  static Future<void> deleteAllFiles() async {
    final List<File> files = await listCompatibleFiles();
    for (final File file in files) {
      if (await _isDataFileCompatible(file)) {
        file.delete();
      }
    }
  }
}

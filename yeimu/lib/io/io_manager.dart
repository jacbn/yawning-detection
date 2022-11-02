import 'package:path_provider/path_provider.dart';
import 'package:yeimu/structure/file_metadata.dart';
import 'dart:io';
import 'dart:convert';

import 'package:yeimu/versioning/version.dart';

class IOManager {
  late FileMetadata metadata;

  static Future<bool> saveData(String data) async {
    final FileMetadata metadata = await _getMetaData();
    final File file = await _getLocalFile('data_${metadata.numFiles}');
    // when writing a file, always make the first line the version it was made with:
    file.writeAsString('${Version.version}\n$data');
    return true;
  }

  static Future<String> loadData(String filename) async {
    // note: no compatibility check, assumes the file has been picked from the list provided by listCompatibleFiles()
    final File file = await _getLocalFile(filename);
    final String str = await file.readAsString();
    return str.substring(1 + str.indexOf('\n')); // remove the version number
  }

  static Future<List<String>> listCompatibleFiles() async {
    final Directory dir = await _localDirectory;
    final List<File> files = dir.listSync().whereType<File>().toList();
    // no async filter, so do manually:
    final List<String> compatibleFileNames = [];
    for (final File file in files) {
      if (await _isDataFileCompatible(file)) {
        compatibleFileNames.add(file.path.split('/').last.split('.').first);
      }
    }
    return compatibleFileNames;
  }

  static Future<bool> _isDataFileCompatible(File file) async {
    if (!file.path.contains('data_')) {
      return false;
    }
    final String firstLine =
        await file.openRead().transform(utf8.decoder).transform(const LineSplitter()).first;
    try {
      return int.parse(firstLine) >= Version.lowestCompatibleVersion;
    } catch (e) {
      return false;
    }
  }

  static Future<FileMetadata> _getMetaData() async {
    final File metadataFile = await _localMetaData;
    if (!await metadataFile.exists()) {
      // create new metadata file
      metadataFile.writeAsString('${Version.version}\n0');
      return FileMetadata(Version.version, 0);
    }
    final String contents = await metadataFile.readAsString();
    final List<String> lines = contents.split('\n');
    return FileMetadata(int.parse(lines[0]), int.parse(lines[1]));
  }

  static Future<Directory> get _localDirectory async {
    // TODO: redundant?
    return await getApplicationDocumentsDirectory();
  }

  static Future<File> get _localMetaData async {
    final Directory dir = await _localDirectory;
    return File('${dir.path}/.metadata');
  }

  static Future<File> _getLocalFile(String filename) async {
    final Directory dir = await _localDirectory;
    return File('${dir.path}/$filename.txt');
  }
}

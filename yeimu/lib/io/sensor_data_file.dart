import 'dart:io';

import 'package:yeimu/structure/sensor_reading.dart';

import '../structure/timestamps.dart';
import '../versioning/version.dart';

class SensorDataFile {
  final String filename;
  final int version;
  final List<SensorReading> data;
  final List<Timestamp> timestamps;
  final int sampleRate;

  SensorDataFile(this.filename, this.version, this.data, this.timestamps, this.sampleRate);

  factory SensorDataFile.from(String filename, String fileContents) {
    final List<String> lines = fileContents.split('\n');
    int version = int.parse(lines[0]);
    // TODO: lines[1] currently reserved for splitting one session into multiple files
    final List<Timestamp> timestamps = [];
    int sampleRate = int.parse(lines[2]);
    int numTimestamps = int.parse(lines[3]);
    for (int i = 4; i < 4 + numTimestamps; i++) {
      List<String> split = lines[i].split(' ');
      timestamps.add(Timestamp.fromEventName(int.parse(split[0]), split[1]));
    }
    final List<SensorReading> data = [];
    for (int i = 4 + numTimestamps; i < lines.length; i++) {
      if (lines[i].isNotEmpty) {
        data.add(SensorReading.fromString(lines[i]));
      }
    }

    return SensorDataFile(filename, version, data, timestamps, sampleRate);
  }

  static Future<bool> save(
    File file,
    List<Timestamp> timestamps,
    List<SensorReading> data,
    int sampleRate,
  ) async {
    // first line is version, second is reserved, third is sample rate, fourth is number of timestamps, then timestamps, then data
    file.writeAsString('${Version.version}\n'
        '\n'
        '$sampleRate\n'
        '${timestamps.length}\n'
        '${timestamps.map((t) => t.toString()).join('\n')}\n'
        '${data.join('\n')}');
    return true;
  }
}

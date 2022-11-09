class SensorReading {
  final List<int> accel;
  final List<int> gyro;

  SensorReading(this.accel, this.gyro);

  @override
  String toString() {
    return '$accel $gyro';
  }

  factory SensorReading.fromString(String string) {
    // unfortunately list to string conversion is printed with spaces between elements, so we instead split by list chars
    final List<String> accelAndGyroLists = string
        .split('[')
        .where((v) => v.isNotEmpty)
        .map((w) => w.replaceAll(RegExp(r'[,\]]'), ''))
        .toList();
    SensorReading reading = SensorReading(
      accelAndGyroLists[0] // accel list
          .split(' ') // split into individual axes
          .where((v) => v.isNotEmpty) // list of list string representations have two commas (??)
          .map((v) => int.parse(v)) // parse each value to int
          .toList(),
      //same again for gyro:
      accelAndGyroLists[1].split(' ').where((v) => v.isNotEmpty).map((v) => int.parse(v)).toList(),
    );

    if (reading.accel.length != 3 || reading.gyro.length != 3) {
      throw Exception('Invalid sensor reading: $reading');
    }

    return reading;
  }

  static String sensorReadingsToString(List<SensorReading> data) {
    return data.map((e) => e.toString()).join('\n');
  }

  static List<SensorReading> stringToSensorReadings(String data) {
    return data.split('\n').map((pair) => SensorReading.fromString(pair)).toList();
  }
}

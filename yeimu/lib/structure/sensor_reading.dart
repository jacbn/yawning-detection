class SensorReading {
  final List<int>? accel;
  final List<int>? gyro;

  SensorReading(this.accel, this.gyro);

  @override
  String toString() {
    return '$accel $gyro';
  }

  static String sensorReadingsToString(List<SensorReading> data) {
    return data.map((e) => e.toString()).join('\n');
  }

  static List<SensorReading> stringToSensorReadings(String data) {
    return data.split('\n').map((pair) {
      final List<String> accelAndGyroLists = pair
          .split('[')
          .where((v) => v.isNotEmpty)
          .map((w) => w.replaceAll(RegExp(r'[,\]]'), ''))
          .toList();
      return SensorReading(
        accelAndGyroLists[0] // accel list
            .substring(1, accelAndGyroLists[0].length) // exclude the square brackets
            .split(' ') // split into individual axes
            .where((v) => v.isNotEmpty) // list of list string representations have two commas (??)
            .map((v) => int.parse(v)) // parse each value to int
            .toList(),
        //same again for gyro:
        accelAndGyroLists[1]
            .substring(1, accelAndGyroLists[1].length)
            .split(' ')
            .where((v) => v.isNotEmpty)
            .map((v) => int.parse(v))
            .toList(),
      );
    }).toList();
  }
}

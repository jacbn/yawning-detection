import 'dart:async';
import 'dart:math';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'package:yeimu/io/sensor_data_file.dart';

import 'io/io_manager.dart';
import 'structure/sensor_reading.dart';
import 'structure/timestamps.dart';

class Results extends StatefulWidget {
  const Results({super.key, required this.data, required this.timestamps});
  final List<SensorReading> data;
  final List<Timestamp> timestamps;

  @override
  State<Results> createState() => _ResultsState();
}

class _ResultsState extends State<Results> {
  List<Color> axisColors = [Colors.red, Colors.green, Colors.blue];
  String? dropdownValue;
  List<VerticalRangeAnnotation> _timestampAnnotations = [];

  Future<List<DropdownMenuItem<String>>> getSavedData() async {
    List<String> files = await IOManager.listCompatibleFilenames();
    return files
        .map((String file) => DropdownMenuItem<String>(value: file, child: Text(file)))
        .toList();
  }

  void updateCurrentFile(String newFile) async {
    SensorDataFile file = await IOManager.loadData(newFile);

    setState(() {
      dropdownValue = newFile;
      widget.data.clear();
      widget.data.addAll(file.data);
      widget.timestamps.clear();
      widget.timestamps.addAll(file.timestamps);

      _timestampAnnotations = file.timestamps
          .map((t) => VerticalRangeAnnotation(
                x1: max(0, t.time.toDouble() - file.data.length / 200),
                x2: min(widget.data.length - 1, t.time.toDouble() + file.data.length / 200),
                color: Colors.black.withOpacity(0.8),
              ))
          .toList();
    });
  }

  void _confirmDeleteAllFiles() {
    // confirm dialog
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Delete all files?'),
          content: const Text('This cannot be undone.'),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                IOManager.deleteAllFiles();
                Navigator.of(context).pop();
              },
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Results'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              ElevatedButton(
                style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Theme.of(context).primaryColor)),
                onPressed: () {
                  IOManager.shareAllFiles();
                },
                child: const Text("Share", style: TextStyle(color: Colors.white)),
              ),
              FutureBuilder(
                future: getSavedData(),
                builder: (BuildContext context, AsyncSnapshot snapshot) {
                  if (snapshot.hasData) {
                    return DropdownButton(
                      value: dropdownValue,
                      items: snapshot.data,
                      onChanged: (String? s) {
                        if (s != null) {
                          updateCurrentFile(s);
                        }
                      },
                    );
                  } else if (snapshot.hasError) {
                    return Text('Error: ${snapshot.error}');
                  } else {
                    return const CircularProgressIndicator();
                  }
                },
              ),
              ElevatedButton(
                style: ButtonStyle(backgroundColor: MaterialStateProperty.all(Colors.red[600])),
                onPressed: () {
                  _confirmDeleteAllFiles();
                },
                child: const Text("Delete All", style: TextStyle(color: Colors.white)),
              ),
            ],
          ),
          Text('Accelerometer Data', style: Theme.of(context).textTheme.headline5),
          Padding(
            padding: const EdgeInsets.only(left: 0, top: 8, right: 8, bottom: 2),
            child: AspectRatio(
              aspectRatio: 1.5,
              child: DecoratedBox(
                decoration: const BoxDecoration(
                  borderRadius: BorderRadius.all(Radius.circular(10)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: LineChart(accelData(widget.data)),
                ),
              ),
            ),
          ),
          Text('Gyroscope Data', style: Theme.of(context).textTheme.headline5),
          Padding(
            padding: const EdgeInsets.only(left: 0, top: 8, right: 8, bottom: 2),
            child: AspectRatio(
              aspectRatio: 1.5,
              child: DecoratedBox(
                decoration: const BoxDecoration(
                  borderRadius: BorderRadius.all(Radius.circular(10)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: LineChart(gyroData(widget.data)),
                ),
              ),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("Legend:", style: Theme.of(context).textTheme.bodyLarge),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(width: 20),
                  legendItem(Colors.red, 'X'),
                  const SizedBox(width: 20),
                  legendItem(Colors.green, 'Y'),
                  const SizedBox(width: 20),
                  legendItem(Colors.blue, 'Z'),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Row legendItem(Color color, String textVal) {
    return Row(
      children: [
        Container(
          width: 20,
          height: 20,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            color: color,
          ),
        ),
        const SizedBox(width: 5),
        Text(textVal),
      ],
    );
  }

  LineChartData gyroData(List<SensorReading> allData) {
    List<LineChartBarData> gyroAxisReadings = getIMUReadings(allData, (v) => v.gyro);
    return makeLineChartData(gyroAxisReadings, allData.length);
  }

  LineChartData accelData(List<SensorReading> allData) {
    List<LineChartBarData> accelAxisReadings = getIMUReadings(allData, (v) => v.accel);
    return makeLineChartData(accelAxisReadings, allData.length);
  }

  List<LineChartBarData> getIMUReadings(List<SensorReading> allData, Function sensorReadingToAxes) {
    // for each IMU field, for each axis (x, y, z), map allData to the list of values for that axis
    List<LineChartBarData> axisReadings = [];
    for (int axis = 0; axis < axisColors.length; axis++) {
      List<FlSpot> values = [];
      for (int i = 0; i < allData.length; i++) {
        List<int> sensorReading = sensorReadingToAxes(allData[i]);
        values.add(FlSpot(i.toDouble(), sensorReading[axis].toDouble()));
      }
      LineChartBarData reading = LineChartBarData(
        spots: values,
        isCurved: true,
        isStrokeCapRound: true,
        color: axisColors[axis],
      );
      axisReadings.add(reading);
    }
    return axisReadings;
  }

  LineChartData makeLineChartData(List<LineChartBarData> data, int numPoints) {
    double minY = (data[0].spots.isEmpty) ? -1 : data.map((e) => e.mostBottomSpot.y).reduce(min);
    double maxY = (data[0].spots.isEmpty) ? 1 : data.map((e) => e.mostTopSpot.y).reduce(max);
    double horizontalInterval = (max(minY.abs(), maxY.abs())) / 4;
    double verticalInterval = max(1, numPoints / 8);
    return LineChartData(
      minX: 0,
      minY: -max(minY.abs(), maxY.abs()),
      maxX: numPoints - 1.0,
      maxY: max(minY.abs(), maxY.abs()),
      gridData: FlGridData(
        show: true,
        drawVerticalLine: true,
        horizontalInterval: horizontalInterval,
        verticalInterval: verticalInterval,
      ),
      lineBarsData: data,
      titlesData: FlTitlesData(
        show: true,
        rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        bottomTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 30,
            interval: verticalInterval,
          ),
        ),
        leftTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 50,
            interval: horizontalInterval,
          ),
        ),
      ),
      rangeAnnotations: RangeAnnotations(
        verticalRangeAnnotations: [
          VerticalRangeAnnotation(
            x1: 0,
            x2: numPoints - 1.0,
            color: Colors.grey.withOpacity(0.1),
          ),
        ].followedBy(_timestampAnnotations).toList(),
      ),
    );
  }
}
